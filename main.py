import io
import numpy as np
import torch
import whisper
import tempfile
from pydub import AudioSegment
from moviepy import VideoFileClip
from pydub.silence import detect_nonsilent
from pydub.playback import play
from subprocess import CalledProcessError, run


def extract_audio(movie_path: str, frame_rate: int = 16000):
    movie = VideoFileClip(movie_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f"{temp_dir}\\temp.file.wav"
        movie.audio.write_audiofile(file_name, codec="pcm_s16le", fps=frame_rate, buffersize=2000)
        movie.close()

        with open(file_name, "rb") as f:
            buffer = io.BytesIO()
            buffer.write(f.read())
            buffer.seek(0)

    return buffer


if __name__ == "__main__":
    video_path = "C:\\Osobné\\HDD\\filmy\\Vlny  2024 CZ.mp4"
    rate = 16000

    aud_buffer = extract_audio(video_path, frame_rate=rate)
    audio = AudioSegment.from_file(aud_buffer, format="wav")

    non_silent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("large", device=device)

    for idx, (start, end) in enumerate(non_silent_ranges):
        if idx >= 1:
            continue
        start = start - 500 if start - 500 > 0 else start
        end = end + 500 if (end + 500) < len(audio) else end
        audio_chunk = audio[start:end]
        chunk_io = io.BytesIO()
        audio_chunk.export(chunk_io, format="wav")
        chunk_io.seek(0)
        chunk_io.name = "temp.wav"
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", chunk_io,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(16000),
            "-"
        ]
        # fmt: on
        try:
            out = run(cmd, capture_output=True, check=True).stdout
            print(np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0)
        except CalledProcessError as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
