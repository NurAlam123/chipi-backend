import io
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

from config import ABS_PATH, SD_CHANNELS, SD_SAMPLE_RATE, WHISPER_MODEL_PATH


class STT:
    DURATION = 10

    def __init__(self) -> None:
        pass

    def record(self) -> None:
        self.audio = sd.rec(
            self.DURATION * SD_SAMPLE_RATE,
            samplerate=SD_SAMPLE_RATE,
            channels=SD_CHANNELS,
            dtype="float32",
        )
        sd.wait()

    def save(self, filepath="record.wav") -> None:
        if not hasattr(self, "audio"):
            raise AttributeError("Audio not found. Record a audio first.")

        sf.write(filepath, self.audio, SD_SAMPLE_RATE)

    def stream(self) -> None:
        if not hasattr(self, "audio"):
            raise AttributeError("Audio not found. Record a audio first.")

        self.audio_file = io.BytesIO()
        sf.write(self.audio_file, self.audio, SD_SAMPLE_RATE, format="WAV")
        self.audio_file.seek(0)

    def playrec(self) -> None:
        if not hasattr(self, "audio"):
            raise AttributeError("Audio not found. Record a audio first.")

        sd.playrec(self.audio, samplerate=SD_SAMPLE_RATE, channels=SD_CHANNELS)
        sd.wait()

    def transcribe(self, audio="", device="cpu") -> str:
        if not hasattr(self, "audio_file") and not audio:
            raise AttributeError(
                "Audio not found. Record a audio first and then stream or provide a file path."
            )

        if not audio:
            audio = self.audio_file
        else:
            audio = ABS_PATH / audio

        model = WhisperModel(
            WHISPER_MODEL_PATH.__str__(), device=device, compute_type="int8"
        )
        segments, _ = model.transcribe(audio.__str__(), beam_size=5, vad_filter=True)

        full_text = " ".join(seg.text for seg in segments)
        return full_text


if __name__ == "__main__":
    stt = STT()

    print("RECORDING...")
    stt.record()
    print("DONE")

    print("PLAYING...")
    stt.playrec()
    print("DONE")

    stt.stream()

    print("TRANSCRIBING...")
    full_text = stt.transcribe()
    print("TEXT: ", full_text)
    print("DONE")


# print("NOISE REDUCING...")
# reduced_noise = nr.reduce_noise(y=audio.flatten(), sr=sample_rate, prop_decrease=1.0)
# print("DONE")
