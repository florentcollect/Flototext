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
        """Load the model in a background thread."""
        if self._loading or self._model_loaded:
            return

        thread = threading.Thread(target=self._load_model, daemon=True)
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
        # _loading is only raised once the worker thread starts, so we cannot
        # read "not loading" as failure until we have seen it go up at least
        # once - otherwise we would give up in the gap right after the call.
        demarre = False
        while time.monotonic() < limite:
            if self._model_loaded:
                return True
            if self._loading:
                demarre = True
            elif demarre:
                # Loading ended without success; _load_model already reported.
                return False
            time.sleep(0.1)
        return self._model_loaded

    def _load_model(self) -> None:
        """Load the ASR backend selected in the configuration."""
        with self._lock:
            if self._model_loaded:
                return

            self._loading = True

        if self._dry_run:
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

            self._backend = backend
            self._model_loaded = True
            self._loading = False
            self._last_used = time.monotonic()
            self._unloaded_when_idle = False

            print("Model loaded successfully")

            if self.on_model_loaded:
                self.on_model_loaded()

        except Exception as e:
            self._loading = False
            error_msg = f"Failed to load model: {e}"
            print(error_msg)

            if self.on_error:
                self.on_error(error_msg)

    def reload_backend(self) -> None:
        """Switch ASR engine: unload the current backend and reload from config.

        Used when the user changes the backend at runtime. Loading happens in a
        background thread; ``on_model_loaded`` fires again when ready.
        """
        self.cleanup()
        self._model_loaded = False
        self._loading = False
        self.load_model_async()

    def transcribe(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """Transcribe audio data to text.

        Args:
            audio_data: Audio data as numpy array.
            sample_rate: Sample rate of the audio.

        Returns:
            TranscriptionResult with the transcribed text.
        """
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
