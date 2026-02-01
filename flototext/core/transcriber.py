"""Transcription module using Qwen3-ASR model."""

import threading
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

from ..config import config
from .localization import localization


@dataclass
class TranscriptionResult:
    """Result of a transcription."""
    text: str
    language: str
    success: bool
    error: Optional[str] = None


class Transcriber:
    """Transcribes audio using Qwen3-ASR model."""

    def __init__(
        self,
        on_model_loaded: Optional[Callable] = None,
        on_error: Optional[Callable[[str], None]] = None
    ):
        """Initialize the transcriber.

        Args:
            on_model_loaded: Callback when model is loaded.
            on_error: Callback when an error occurs.
        """
        self.on_model_loaded = on_model_loaded
        self.on_error = on_error

        self._model = None
        self._model_loaded = False
        self._loading = False
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        """Check if model is loaded and ready."""
        return self._model_loaded

    @property
    def is_loading(self) -> bool:
        """Check if model is currently loading."""
        return self._loading

    def load_model_async(self) -> None:
        """Load the model in a background thread."""
        if self._loading or self._model_loaded:
            return

        thread = threading.Thread(target=self._load_model, daemon=True)
        thread.start()

    def _load_model(self) -> None:
        """Load the Qwen3-ASR model."""
        with self._lock:
            if self._model_loaded:
                return

            self._loading = True

        try:
            import torch
            from qwen_asr import Qwen3ASRModel

            print(f"Loading model: {config.model.model_name}")

            # Determine dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(config.model.dtype, torch.bfloat16)

            # Load model using qwen-asr API
            self._model = Qwen3ASRModel.from_pretrained(
                config.model.model_name,
                dtype=dtype,
                device_map=config.model.device,
                max_inference_batch_size=1,
                max_new_tokens=config.model.max_new_tokens,
            )

            self._model_loaded = True
            self._loading = False

            print("Model loaded successfully")

            if self.on_model_loaded:
                self.on_model_loaded()

        except Exception as e:
            self._loading = False
            error_msg = f"Failed to load model: {e}"
            print(error_msg)

            if self.on_error:
                self.on_error(error_msg)

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

        try:
            import torch

            # Ensure audio is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize audio
            max_val = np.abs(audio_data).max()
            if max_val > 0:
                # Normalize to [-1, 1] range for better ASR performance
                audio_data = audio_data / max_val

            # Transcribe using Qwen3-ASR
            # The model accepts (np.ndarray, sample_rate) tuples
            results = self._model.transcribe(
                audio=(audio_data, sample_rate),
                language=localization.asr_language,
            )

            # Get the transcription result
            if results and len(results) > 0:
                transcription = results[0].text.strip()
                detected_language = results[0].language or localization.asr_language
            else:
                transcription = ""
                detected_language = localization.asr_language

            # Clear CUDA cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return TranscriptionResult(
                text=transcription,
                language=detected_language,
                success=True
            )

        except torch.cuda.OutOfMemoryError:
            import torch
            torch.cuda.empty_cache()
            error_msg = localization.get("errors.gpu_oom")
            if self.on_error:
                self.on_error(error_msg)
            return TranscriptionResult(
                text="",
                language=localization.asr_language,
                success=False,
                error=error_msg
            )

        except Exception as e:
            error_msg = localization.get("errors.transcription_error", error=str(e))
            print(error_msg)
            if self.on_error:
                self.on_error(error_msg)
            return TranscriptionResult(
                text="",
                language=localization.asr_language,
                success=False,
                error=error_msg
            )

    def cleanup(self) -> None:
        """Clean up model resources."""
        try:
            import torch

            if self._model is not None:
                del self._model
                self._model = None

            self._model_loaded = False

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error during cleanup: {e}")
