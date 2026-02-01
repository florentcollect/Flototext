"""Audio recording module using sounddevice."""

import numpy as np
import sounddevice as sd
import threading
import time
from typing import Optional, Callable
from dataclasses import dataclass

from ..config import config


@dataclass
class RecordingResult:
    """Result of an audio recording."""
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    is_valid: bool


class AudioRecorder:
    """Records audio from the microphone."""

    def __init__(
        self,
        sample_rate: int = None,
        channels: int = None,
        on_start: Optional[Callable] = None,
        on_stop: Optional[Callable] = None
    ):
        """Initialize the audio recorder.

        Args:
            sample_rate: Audio sample rate (default from config).
            channels: Number of audio channels (default from config).
            on_start: Callback when recording starts.
            on_stop: Callback when recording stops.
        """
        self.sample_rate = sample_rate or config.audio.sample_rate
        self.channels = channels or config.audio.channels
        self.on_start = on_start
        self.on_stop = on_stop

        self._recording = False
        self._audio_data: list = []
        self._stream: Optional[sd.InputStream] = None
        self._lock = threading.Lock()
        self._data_lock = threading.Lock()  # Separate lock for audio data access
        self._start_time: float = 0

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status):
        """Callback for audio stream."""
        if status:
            print(f"Audio status: {status}")
        if self._recording:
            with self._data_lock:
                self._audio_data.append(indata.copy())

    def start_recording(self) -> bool:
        """Start recording audio.

        Returns:
            True if recording started successfully.
        """
        with self._lock:
            if self._recording:
                return False

            try:
                self._audio_data = []
                self._start_time = time.time()

                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype=config.audio.dtype,
                    callback=self._audio_callback
                )
                self._stream.start()
                self._recording = True

                if self.on_start:
                    self.on_start()

                return True

            except Exception as e:
                print(f"Error starting recording: {e}")
                self._recording = False
                return False

    def stop_recording(self) -> Optional[RecordingResult]:
        """Stop recording and return the audio data.

        Returns:
            RecordingResult with the recorded audio, or None if error.
        """
        with self._lock:
            if not self._recording:
                return None

            try:
                self._recording = False
                duration = time.time() - self._start_time

                if self._stream:
                    self._stream.stop()
                    self._stream.close()
                    self._stream = None

                if self.on_stop:
                    self.on_stop()

                # Copy audio data under lock to prevent race condition
                with self._data_lock:
                    if not self._audio_data:
                        return RecordingResult(
                            audio_data=np.array([]),
                            sample_rate=self.sample_rate,
                            duration=0,
                            is_valid=False
                        )
                    # Concatenate all audio chunks
                    audio_data = np.concatenate(self._audio_data, axis=0)

                # Flatten to mono if needed
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.flatten()

                # Check minimum duration
                is_valid = duration >= config.audio.min_duration

                return RecordingResult(
                    audio_data=audio_data,
                    sample_rate=self.sample_rate,
                    duration=duration,
                    is_valid=is_valid
                )

            except Exception as e:
                print(f"Error stopping recording: {e}")
                return None

    def get_input_devices(self) -> list:
        """Get list of available input devices.

        Returns:
            List of input device info dictionaries.
        """
        devices = sd.query_devices()
        input_devices = []

        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })

        return input_devices

    def set_input_device(self, device_id: int) -> bool:
        """Set the input device to use.

        Args:
            device_id: The device ID to use.

        Returns:
            True if device was set successfully.
        """
        try:
            sd.default.device[0] = device_id
            return True
        except Exception as e:
            print(f"Error setting input device: {e}")
            return False

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self._recording = False
