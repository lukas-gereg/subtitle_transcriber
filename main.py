import io
import tempfile
from pydub import AudioSegment
from moviepy import VideoFileClip
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
    video_path = "C:\\Osobné\\HDD\\filmy\\Vlny  2024 CZ.mp4"
    rate = 16000

    aud_buffer = extract_audio(video_path, frame_rate=rate)
    audio = AudioSegment.from_file(aud_buffer, format="wav")

    non_silent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)

    for idx, (start, end) in enumerate(non_silent_ranges):
        print(idx)
        audio_chunk = audio[start:end]
        chunk_io = io.BytesIO()
        audio_chunk.export(chunk_io, format="wav")