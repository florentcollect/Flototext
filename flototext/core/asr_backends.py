"""ASR backend abstraction.

Two interchangeable speech-to-text engines sit behind a common interface so the
rest of the app (and the benchmark script) never has to care which one is loaded:

- ``QwenBackend``       -> Qwen/Qwen3-ASR-1.7B via the ``qwen_asr`` library.
- ``CanaryOnnxBackend`` -> NVIDIA Canary 1B v2 via ``onnx_asr`` (ONNX Runtime).

Heavy imports (torch, qwen_asr, onnx_asr) are deferred to ``load()`` so importing
this module stays cheap and a missing optional dependency only fails the backend
that actually needs it.
"""

import math
import re

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

    # Left to its defaults, ONNX Runtime reserves far more VRAM than Canary
    # needs: measured at 8.7 GB for a model whose weights are 3.8 GB. Three
    # defaults account for the gap, and each is pinned below.
    #
    #   arena_extend_strategy: kNextPowerOfTwo doubles the arena every time it
    #       runs short and never returns the memory. kSameAsRequested grows it
    #       by what was actually asked for.
    #   cudnn_conv_use_max_workspace: "1" since ORT 1.14 - convolutions get the
    #       largest workspace cuDNN will take, traded against a speed we do not
    #       need for 20 s of speech.
    #   cudnn_conv_algo_search: EXHAUSTIVE benchmarks every convolution
    #       algorithm at load time, allocating each one's workspace to do it.
    #
    # gpu_mem_limit caps each ONNX session, not the process: Canary opens one
    # session for the encoder and one for the decoder, so the process ceiling is
    # twice this value. 4 GiB is the floor that still fits the encoder's 3.06 GiB
    # of weights with working room - measured over 16 back-to-back inferences at
    # lengths up to the 20 s MAX_SEGMENT_SECONDS maximum, zero allocation
    # failures, VRAM settling at 4.65 GB. Going lower risks failing at session
    # creation, when the encoder's initializers are copied to the device.
    #
    # The cap alone is not enough: the arena keeps every block it ever carved,
    # sized for each new input length, and never merges them back. With the
    # weights holding 3.06 of the 4 GiB, the room left fills up after some
    # thirty distinct lengths, and from then on every segment over ~15 s asks
    # for 49-65 MB the arena cannot find: 20 refusals out of 149 back-to-back
    # clips (2026-11-19). _install_arena_shrinkage() below is the fix.
    #
    # do_copy_in_default_stream already defaults to "1" and is left out on
    # purpose: setting it changes nothing.
    CUDA_PROVIDER_OPTIONS = {
        "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": 4 * 1024 ** 3,
        "cudnn_conv_algo_search": "HEURISTIC",
        "cudnn_conv_use_max_workspace": "0",
    }

    def __init__(self) -> None:
        self._model = None
        self._releasing_decoder = None

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
            # onnx_asr passes this straight to InferenceSession, which accepts
            # a (provider, options) tuple alongside plain provider names.
            providers = [
                ("CUDAExecutionProvider", self.CUDA_PROVIDER_OPTIONS),
                "CPUExecutionProvider",
            ]

        # onnx-asr's Resampler builds one InferenceSession per source sample rate
        # (8/11/22/24/32/44.1/48 kHz) and only excludes TensorRT, so all seven
        # land on CUDA with their own arena and cuBLAS/cuDNN handles. None of
        # them ever runs: we always feed 16 kHz, and Resampler.__call__ returns
        # its input untouched when the rate already matches. Keep them on CPU.
        resampler_config = {"providers": ["CPUExecutionProvider"]}

        model_name = self._model_name()
        print(f"Loading {self.name} ONNX model: {model_name} "
              f"(providers={'CUDA (capped)' if providers else 'default'})")
        # Downloads the ONNX weights from Hugging Face on first run, then caches.
        if providers:
            self._model = onnx_asr.load_model(
                model_name,
                providers=providers,
                resampler_config=resampler_config,
            )
        else:
            self._model = onnx_asr.load_model(
                model_name,
                resampler_config=resampler_config,
            )
        if providers:
            self._install_arena_shrinkage()
        print(f"{self.name} model loaded successfully")

    def _install_arena_shrinkage(self) -> None:
        """Hand the GPU arena's free blocks back after each dictation.

        ONNX Runtime can shrink an arena at the end of a run, freeing every
        block not in use, when the run carries the
        memory.enable_memory_arena_shrinkage option. onnx-asr calls run()
        without options, so its sessions are wrapped:

        - the encoder runs once per segment and shrinks every time;
        - the decoder runs once per token, and shrinking on each step cost
          +130 ms a dictation, so it shrinks once at the end of transcribe(),
          by replaying the segment's first step.

        Measured over 149 back-to-back clips (2026-11-19): 20 refusals -> 0,
        VRAM between dictations 6.2 GB -> 4.16 GB (the level right after
        loading), median latency +85 ms, WER unchanged. The cap stays as a
        guard. Relies on onnx-asr internals (asr._encoder, asr._decoder,
        decoder_mems): if they move, this is skipped and the arena grows as
        before, with Transcriber's reload-and-retry as the net.
        """
        asr = getattr(self._model, "asr", None)
        encoder = getattr(asr, "_encoder", None)
        if encoder is None:
            print("Arena shrinkage unavailable: onnx-asr has no asr._encoder")
            return

        run_options = _shrink_run_options()
        asr._encoder = _ShrinkingSession(encoder, run_options)

        # Parakeet has a joint decoder instead, with a small fixed footprint.
        decoder = getattr(asr, "_decoder", None)
        if decoder is not None and _ReleasingDecoder.FIRST_STEP_INPUT in {
                i.name for i in decoder.get_inputs()}:
            self._releasing_decoder = asr._decoder = _ReleasingDecoder(decoder, run_options)

    def _release_working_memory(self) -> None:
        if self._releasing_decoder is None:
            return
        try:
            self._releasing_decoder.release()
        except Exception as e:
            # Housekeeping only: the text is already decoded.
            print(f"Could not shrink the decoder arena: {e}")

    def _model_name(self) -> str:
        return config.model.canary_model_name

    def _recognize(self, audio, sample_rate, language_code):
        return self._model.recognize(
            audio, sample_rate=sample_rate, language=language_code
        )

    def transcribe(self, audio, sample_rate, language_label, language_code):
        try:
            return self._transcribe_segments(audio, sample_rate, language_code)
        finally:
            self._release_working_memory()

    def _transcribe_segments(self, audio, sample_rate, language_code):
        max_samples = int(self.MAX_SEGMENT_SECONDS * sample_rate)

        if len(audio) <= max_samples:
            text = self._recognize(audio, sample_rate, language_code)
            return _clean(text), language_code

        # Long recording: split into equal windows and stitch the transcripts.
        # Equal, not fixed: a fixed 20 s stride left a tail of any length, down
        # to a few samples, and under 320 samples (20 ms) onnx-asr's feature
        # extractor divides by zero; the NaN features sent the decoder running
        # until its arena failed (BFCArena on /_decoder/Concat_11, 4 times in
        # the log). Equal windows are always over half the limit.
        count = math.ceil(len(audio) / max_samples)
        size = math.ceil(len(audio) / count)
        parts = []
        for start in range(0, len(audio), size):
            chunk = audio[start:start + size]
            part = self._recognize(chunk, sample_rate, language_code)
            if part:
                parts.append(part.strip())
        return _clean(" ".join(parts)), language_code

    def cleanup(self) -> None:
        # onnxruntime sessions are released when the model is dereferenced.
        self._model = None
        self._releasing_decoder = None


_SHRINK_ARENA = "memory.enable_memory_arena_shrinkage"


def _shrink_run_options():
    import onnxruntime as ort

    run_options = ort.RunOptions()
    run_options.add_run_config_entry(_SHRINK_ARENA, "gpu:0")
    return run_options


class _ShrinkingSession:
    """An InferenceSession whose every run shrinks the GPU arena afterwards."""

    def __init__(self, session, run_options):
        self._session = session
        self._run_options = run_options

    def run(self, output_names, input_feed, run_options=None):
        return self._session.run(output_names, input_feed, self._run_options)

    def __getattr__(self, name):
        return getattr(self._session, name)


class _ReleasingDecoder:
    """Canary's decoder, able to shrink its arena once a dictation is done.

    A shrink happens only at the end of a run, and the last step of the
    decoding loop is not known in advance. So the first step of each segment,
    the one with empty decoder_mems, is kept and replayed with the shrink
    option by release(): one extra step instead of one shrink per token.
    """

    FIRST_STEP_INPUT = "decoder_mems"

    def __init__(self, session, run_options):
        self._session = session
        self._run_options = run_options
        self._first_step = None

    def run(self, output_names, input_feed, run_options=None):
        mems = input_feed.get(self.FIRST_STEP_INPUT)
        if mems is not None and mems.shape[2] == 0:
            self._first_step = (output_names, input_feed)
        return self._session.run(output_names, input_feed, run_options)

    def release(self) -> None:
        step, self._first_step = self._first_step, None
        if step is not None:
            self._session.run(step[0], step[1], self._run_options)

    def __getattr__(self, name):
        return getattr(self._session, name)


_UNK_RE = re.compile(r"\s*<unk>\s*")


def _clean(text) -> str:
    """Drop the <unk> tokens the NeMo decoders emit for unknown pieces.

    They were pasted as is ("Fletch<unk> Friends", "répéter<unk> il<unk>").
    Removing one leaves a single space where it sat between two words.
    """
    text = _UNK_RE.sub(" ", text or "")
    return re.sub(r"\s{2,}", " ", text).strip()


class ParakeetOnnxBackend(CanaryOnnxBackend):
    """NVIDIA Parakeet TDT 0.6B v3 backend via onnx-asr (ONNX Runtime).

    Same loading path as Canary (CUDA pinned, arena capped, resampler on CPU):
    only the model and the recognize() call differ. Parakeet v3 detects the
    language on its own and takes no language argument.
    """

    name = "parakeet"

    def _model_name(self) -> str:
        return config.model.parakeet_model_name

    def _recognize(self, audio, sample_rate, language_code):
        return self._model.recognize(audio, sample_rate=sample_rate)


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
    if name == "parakeet":
        return ParakeetOnnxBackend()
    if name != "qwen":
        print(f"Unknown ASR backend '{name}', falling back to Qwen")
    return QwenBackend()
