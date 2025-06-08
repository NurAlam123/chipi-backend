from collections.abc import Iterator
from piper.voice import PiperVoice
import numpy as np
import sounddevice as sd
import wave
import soundfile as sf

from config import ABS_PATH, TTS_MODEL


class TTS:
    def __init__(self) -> None:
        self.voice = PiperVoice.load(TTS_MODEL)

    def stream(self, text: str):
        self.audio = self.voice.synthesize_stream_raw(text)

    def save(self, text: str, filepath: str = "output-tts.wav"):
        full_path = ABS_PATH / filepath
        with wave.open(full_path.__str__(), mode="wb") as wav:
            self.voice.synthesize(text, wav)
        self.audio = full_path

    def say(self):
        if not hasattr(self, "audio"):
            raise AttributeError("Audio not found. Either stream the audio or save it")

        if isinstance(self.audio, Iterator):
            for i in self.audio:
                audio = np.frombuffer(i, dtype=np.int16).astype(np.float32) / 32768.0

                sd.play(audio, samplerate=22050)
                sd.wait()
        else:
            audio_data, samplerate = sf.read(self.audio)
            sd.play(audio_data, samplerate)
            sd.wait()


if __name__ == "__main__":
    text = "HELLO! THIS IS PIPER"
    tts = TTS()
    tts.save(text)
    tts.say()
