"""Transcription module.

Thin orchestration layer around a pluggable ASR backend (Qwen or Canary). The
backend is selected from ``config.model.backend``; this class keeps the audio
normalization, dry-run handling, OOM recovery and the public
``transcribe(audio_data, sample_rate) -> TranscriptionResult`` contract stable
regardless of which engine is loaded.
"""

import threading
import time
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

from ..config import config
from .localization import localization
from .asr_backends import create_backend, BaseASRBackend


# Messages that mean the engine ran out of device memory rather than failed on
# the audio. Seen in the log: "BFCArena ... Failed to allocate memory" from the
# ONNX arena, "CUBLAS failure 3" when another program had filled the GPU.
_ALLOCATION_FAILURE_MARKERS = (
    "bfcarena",
    "failed to allocate memory",
    "cublas failure",
    "cuda failure",
    "out of memory",
)


def _is_allocation_failure(error: Optional[str]) -> bool:
    """Whether an engine error is a device-memory failure worth one reload."""
    text = (error or "").lower()
    return any(marker in text for marker in _ALLOCATION_FAILURE_MARKERS)


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    language: str
    success: bool
    error: Optional[str] = None


class Transcriber:
    """Transcribes audio using the configured ASR backend."""

    def __init__(
        self,
        on_model_loaded: Optional[Callable] = None,
        on_error: Optional[Callable[[str], None]] = None,
        dry_run: Optional[bool] = None
    ):
        """Initialize the transcriber.

        Args:
            on_model_loaded: Callback when model is loaded.
            on_error: Callback when an error occurs.
            dry_run: Whether to skip model loading and return sample text.
        """
        self.on_model_loaded = on_model_loaded
        self.on_error = on_error
        self._dry_run = config.model.dry_run if dry_run is None else dry_run

        self._backend: Optional[BaseASRBackend] = None
        self._model_loaded = False
        self._loading = False
        self._lock = threading.Lock()

        # Guards the backend itself, so the idle watchdog cannot unload a model
        # that a transcription is still running on. Distinct from _lock, which
        # only serialises loading: taking _lock here would deadlock reload.
        self._engine_lock = threading.RLock()
        self._last_used = time.monotonic()
        # True once the idle watchdog has released the model, so that pressing
        # the hotkey knows this is a reloadable state and not a failed load.
        self._unloaded_when_idle = False
        # Bumped by reload_backend(). A load that finishes under an older
        # generation was superseded (the user picked another engine meanwhile):
        # it releases what it built instead of installing it over the new one.
        self._generation = 0

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model_loaded

    @property
    def is_loading(self) -> bool:
        """Check if model is currently loading."""
        return self._loading

    @property
    def was_unloaded_when_idle(self) -> bool:
        """Whether the model was released after sitting idle."""
        return self._unloaded_when_idle

    @property
    def backend_name(self) -> Optional[str]:
        """Name of the currently loaded backend, if any."""
        return self._backend.name if self._backend else None

    def load_model_async(self) -> None:
        """Load the model in a background thread.

        ``_loading`` is raised here, before the thread starts, and under the
        lock: two callers cannot both start a load, and a waiter never sees
        "not loading" in the gap before the worker runs.
        """
        with self._lock:
            if self._loading or self._model_loaded:
                return
            self._loading = True
            generation = self._generation

        thread = threading.Thread(target=self._load_model, args=(generation,), daemon=True)
        thread.start()

    def unload_if_idle(self, idle_seconds: float) -> bool:
        """Release the model when it has not transcribed for a while.

        The ONNX arena only ever grows: every output length the autoregressive
        decoder has not seen before extends it by a block that is never handed
        back. Dropping the model between working sessions is what actually
        returns that VRAM to the system.

        Skips silently when a transcription is in flight - the watchdog calls
        this on a timer, so the next tick will do.

        Returns:
            True if the model was unloaded by this call.
        """
        if idle_seconds <= 0 or not self._model_loaded or self._loading:
            return False
        if time.monotonic() - self._last_used < idle_seconds:
            return False

        if not self._engine_lock.acquire(blocking=False):
            return False
        try:
            # Re-check under the lock: a transcription may have just finished
            # and refreshed _last_used while we were taking it.
            if not self._model_loaded or time.monotonic() - self._last_used < idle_seconds:
                return False
            minutes = idle_seconds / 60
            print(f"Unloading ASR model after {minutes:.0f} min idle (frees GPU memory)")
            self.cleanup()
            self._unloaded_when_idle = True
            return True
        finally:
            self._engine_lock.release()

    def ensure_loaded(self, timeout: float = 120.0) -> bool:
        """Load the model if needed and wait until it is usable.

        Called on the transcription path so a hotkey press after an idle unload
        just works, at the cost of the reload it triggers.

        Returns:
            True if the model is ready.
        """
        if self._model_loaded:
            return True

        self.load_model_async()
        limite = time.monotonic() + timeout
        while time.monotonic() < limite:
            if self._model_loaded:
                return True
            if not self._loading:
                # Loading ended without success; _load_model already reported.
                return False
            time.sleep(0.05)
        return self._model_loaded

    def _load_model(self, generation: Optional[int] = None) -> None:
        """Load the ASR backend selected in the configuration.

        Args:
            generation: The reload generation this load belongs to; None when
                called directly, meaning "the current one".
        """
        with self._lock:
            if self._model_loaded:
                self._loading = False
                return
            if generation is None:
                generation = self._generation
            self._loading = True

        if self._dry_run:
            with self._lock:
                self._model_loaded = True
                self._loading = False
                self._last_used = time.monotonic()
                self._unloaded_when_idle = False
            print("Dry-run mode enabled; skipping ASR model load")
            if self.on_model_loaded:
                self.on_model_loaded()
            return

        try:
            backend = create_backend(config.model.backend)
            print(f"Loading ASR backend: {backend.name}")
            backend.load()
        except Exception as e:
            with self._lock:
                stale = generation != self._generation
                if not stale:
                    self._loading = False
            if stale:
                return
            error_msg = f"Failed to load model: {e}"
            print(error_msg)
            if self.on_error:
                self.on_error(error_msg)
            return

        with self._lock:
            if generation != self._generation:
                stale = True
            else:
                stale = False
                with self._engine_lock:
                    previous, self._backend = self._backend, backend
                    self._model_loaded = True
                self._loading = False
                self._last_used = time.monotonic()
                self._unloaded_when_idle = False

        if stale:
            # The user switched engine while this one was loading: keep the
            # newer choice and hand this model's VRAM straight back.
            print(f"Discarding superseded {backend.name} load")
            backend.cleanup()
            return
        if previous is not None and previous is not backend:
            previous.cleanup()

        print("Model loaded successfully")
        if self.on_model_loaded:
            self.on_model_loaded()

    def reload_backend(self) -> None:
        """Switch ASR engine: unload the current backend and reload from config.

        Used when the user changes the backend at runtime. Loading happens in a
        background thread; ``on_model_loaded`` fires again when ready. A load
        still running for the previous choice is superseded and cleans up after
        itself.
        """
        with self._lock:
            self._generation += 1
            self._loading = False
        self.cleanup()
        self.load_model_async()

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """Transcribe audio data to text.

        An allocation failure inside the engine (ONNX arena exhausted, cuBLAS
        unable to get memory because something else filled the GPU) is retried
        once on a freshly loaded model: a reload starts from an empty arena,
        and the dictation is otherwise lost.

        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            TranscriptionResult with the transcribed text.
        """
        result = self._transcribe_once(audio_data, sample_rate)
        if result.success or not _is_allocation_failure(result.error):
            return result

        print(f"Engine allocation failure, reloading the model and retrying once: {result.error}")
        self.cleanup()
        if not self.ensure_loaded():
            return result
        return self._transcribe_once(audio_data, sample_rate)

    def _transcribe_once(self, audio_data: np.ndarray, sample_rate: int) -> TranscriptionResult:
        """One transcription attempt on the current engine."""
        if not self._model_loaded:
            return TranscriptionResult(
                text="",
                language=localization.asr_language,
                success=False,
                error=localization.get("errors.model_not_loaded")
            )

        if self._dry_run:
            return TranscriptionResult(
                text=config.model.dry_run_text,
                language=localization.asr_language,
                success=True
            )

        # torch is only needed for CUDA cache management (Qwen backend); the
        # ONNX backend must keep working without it.
        try:
            import torch
        except ImportError:
            torch = None

        try:
            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize audio
            max_val = np.abs(audio_data).max()
            if max_val > 0:
                # Normalize to [-1, 1] range for better ASR performance
                audio_data = audio_data / max_val

            # Delegate to the active backend. Each backend uses the language form
            # it understands (Qwen: "French", Canary: "fr"). The lock keeps the
            # idle watchdog from unloading the backend mid-inference.
            with self._engine_lock:
                self._last_used = time.monotonic()
                if self._backend is None:
                    return TranscriptionResult(
                        text="",
                        language=localization.asr_language,
                        success=False,
                        error=localization.get("errors.model_not_loaded")
                    )
                transcription, detected_language = self._backend.transcribe(
                    audio_data,
                    sample_rate,
                    localization.asr_language,
                    localization.asr_language_code,
                )
                self._last_used = time.monotonic()

            # Clear CUDA cache to free memory
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()

            return TranscriptionResult(
                text=transcription,
                language=detected_language or localization.asr_language,
                success=True
            )

        except Exception as e:
            if torch is not None and isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                return TranscriptionResult(
                    text="",
                    language=localization.asr_language,
                    success=False,
                    error=localization.get("errors.gpu_oom")
                )

            error_msg = localization.get("errors.transcription_error", error=str(e))
            print(error_msg)
            return TranscriptionResult(
                text="",
                language=localization.asr_language,
                success=False,
                error=error_msg
            )

    def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            with self._engine_lock:
                if self._backend is not None:
                    self._backend.cleanup()
                    self._backend = None

                self._model_loaded = False

            try:
                import torch
            except ImportError:
                return
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during cleanup: {e}")
