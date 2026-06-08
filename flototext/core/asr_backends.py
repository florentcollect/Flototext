"""ASR backend abstraction.

Two interchangeable speech-to-text engines sit behind a common interface so the
rest of the app (and the benchmark script) never has to care which one is loaded:

- ``QwenBackend``       -> Qwen/Qwen3-ASR-1.7B via the ``qwen_asr`` library.
- ``CanaryOnnxBackend`` -> NVIDIA Canary 1B v2 via ``onnx_asr`` (ONNX Runtime).

Heavy imports (torch, qwen_asr, onnx_asr) are deferred to ``load()`` so importing
this module stays cheap and a missing optional dependency only fails the backend
that actually needs it.
"""

import numpy as np
from typing import Optional, Tuple

from ..config import config


class BaseASRBackend:
    """Common interface every ASR engine must implement."""

    name: str = "base"

    def load(self) -> None:
        """Load the model into memory. May raise on failure."""
        raise NotImplementedError

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        language_label: str,
        language_code: str,
    ) -> Tuple[str, str]:
        """Transcribe a mono float32 waveform.

        Args:
            audio: Audio samples, mono, float32, normalized to [-1, 1].
            sample_rate: Sample rate of ``audio`` (typically 16000).
            language_label: Full language name (e.g. "French") -- used by Qwen.
            language_code: ISO short code (e.g. "fr") -- used by Canary.

        Returns:
            Tuple of (transcribed_text, detected_language).
        """
        raise NotImplementedError

    def cleanup(self) -> None:
        """Free model resources. Safe to call multiple times."""
        pass


class QwenBackend(BaseASRBackend):
    """Qwen3-ASR backend (the historical default engine)."""

    name = "qwen"

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        import torch
        from qwen_asr import Qwen3ASRModel

        print(f"Loading Qwen model: {config.model.model_name}")

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(config.model.dtype, torch.bfloat16)

        self._model = Qwen3ASRModel.from_pretrained(
            config.model.model_name,
            dtype=dtype,
            device_map=config.model.device,
            max_inference_batch_size=1,
            max_new_tokens=config.model.max_new_tokens,
        )
        print("Qwen model loaded successfully")

    def transcribe(self, audio, sample_rate, language_label, language_code):
        # Qwen accepts (np.ndarray, sample_rate) tuples and a full language name.
        results = self._model.transcribe(
            audio=(audio, sample_rate),
            language=language_label,
        )
        if results and len(results) > 0:
            return results[0].text.strip(), (results[0].language or language_label)
        return "", language_label

    def cleanup(self) -> None:
        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class CanaryOnnxBackend(BaseASRBackend):
    """NVIDIA Canary 1B v2 backend via onnx-asr (ONNX Runtime)."""

    name = "canary"

    # Canary's ONNX export handles roughly 20-30 s per call. We chunk anything
    # longer into <= MAX_SEGMENT_SECONDS windows and concatenate the results.
    # A proper VAD split (onnx_asr .with_vad()) is a future improvement.
    MAX_SEGMENT_SECONDS = 20.0

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        # Register torch's CUDA runtime DLLs so onnxruntime can build the CUDA
        # provider (must happen before the provider DLL is loaded).
        _ensure_cuda_dlls()

        import onnx_asr
        import onnxruntime as ort

        # onnx-asr defaults to a provider list that starts with TensorRT; when
        # TensorRT isn't installed, ONNX Runtime falls back all the way to CPU
        # instead of CUDA. Pin CUDA explicitly (with CPU fallback) so Canary runs
        # on the GPU like Qwen does.
        available = ort.get_available_providers()
        providers = None
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        print(f"Loading Canary ONNX model: {config.model.canary_model_name} "
              f"(providers={providers or 'default'})")
        # Downloads the ONNX weights from Hugging Face on first run, then caches.
        if providers:
            self._model = onnx_asr.load_model(config.model.canary_model_name, providers=providers)
        else:
            self._model = onnx_asr.load_model(config.model.canary_model_name)
        print("Canary model loaded successfully")

    def transcribe(self, audio, sample_rate, language_label, language_code):
        max_samples = int(self.MAX_SEGMENT_SECONDS * sample_rate)

        if len(audio) <= max_samples:
            text = self._model.recognize(
                audio, sample_rate=sample_rate, language=language_code
            )
            return (text or "").strip(), language_code

        # Long recording: split into fixed windows and stitch the transcripts.
        parts = []
        for start in range(0, len(audio), max_samples):
            chunk = audio[start:start + max_samples]
            part = self._model.recognize(
                chunk, sample_rate=sample_rate, language=language_code
            )
            if part:
                parts.append(part.strip())
        return " ".join(parts).strip(), language_code

    def cleanup(self) -> None:
        # onnxruntime sessions are released when the model is dereferenced.
        self._model = None


_CUDA_DLLS_REGISTERED = False


def _ensure_cuda_dlls() -> None:
    """Make ONNX Runtime find the CUDA 12 / cuDNN 9 runtime bundled with torch.

    onnxruntime-gpu needs cublasLt64_12.dll, cudnn64_9.dll, etc. on the DLL
    search path. The torch CUDA wheel already ships them in torch/lib; without
    this, onnxruntime fails to create CUDAExecutionProvider and silently falls
    back to CPU. Registering torch's lib dir lets Canary run on the GPU. No-op on
    non-Windows or if torch isn't present.
    """
    global _CUDA_DLLS_REGISTERED
    if _CUDA_DLLS_REGISTERED:
        return
    try:
        import os
        if not hasattr(os, "add_dll_directory"):  # not Windows
            _CUDA_DLLS_REGISTERED = True
            return
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
    except Exception as e:
        print(f"Could not register torch CUDA DLLs for onnxruntime: {e}")
    finally:
        _CUDA_DLLS_REGISTERED = True


def create_backend(name: str) -> BaseASRBackend:
    """Instantiate the backend matching ``name`` (defaults to Qwen)."""
    if name == "canary":
        return CanaryOnnxBackend()
    return QwenBackend()
