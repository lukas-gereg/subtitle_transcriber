import io
import os
import torch
import tempfile
from pydub import AudioSegment
from moviepy import VideoFileClip
from transformers import pipeline
from pydub.silence import detect_nonsilent
from pydub.playback import play


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
    audio_path = os.path.join(os.getcwd(), "movie_audio.wav")
    rate = 16000

    # aud_buffer = extract_audio(video_path, frame_rate=rate)
    audio = AudioSegment.from_file(audio_path, format="wav")

    non_silent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        chunk_length_s=30,
        device=device,
    )

    for idx, (start, end) in enumerate(non_silent_ranges):
        start = start - 500 if start - 500 > 0 else start
        end = end + 500 if (end + 500) < len(audio) else end

        audio_chunk = audio[start:end]
        chunk_io = io.BytesIO()

        audio_chunk.export(chunk_io, format="wav")
        chunk_io.seek(0)

        prediction = pipe(chunk_io, batch_size=8, return_timestamps=True, return_language=True)

from datetime import timedelta

timeList = ['0:00:00', '0:00:15', '9:30:56', '21:00:00']
total_time = sum(timedelta(hours=int(ms[0]), minutes=int(ms[1]), seconds=int(ms[2])) for t in timeList for ms in [t.split(":")],
        timedelta())