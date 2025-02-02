import io
import os
import torch
import tempfile
from pydub import AudioSegment
from moviepy import VideoFileClip
import torchaudio
from transformers import pipeline
from pydub.silence import detect_nonsilent
from pydub.playback import play
from demucs.pretrained import get_model
from demucs.apply import apply_model
import numpy as np
from denoiser import pretrained


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

def denoise_audio(audio: AudioSegment, device: torch.device):
    demucs_model = get_model("htdemucs_ft").to(device)
    data, sr = audiosegment_to_tensor(audio)

    with torch.no_grad():
        sources = apply_model(demucs_model, data.unsqueeze(0), device=device)

    speech_data = sources[3]
    denoiser_model = pretrained.dns64().to(device)

    with torch.no_grad():
        denoised_vocals = denoiser_model(speech_data.unsqueeze(0)).squeeze(0)

    if sr != audio.frame_rate:
        denoised_vocals = torchaudio.transforms.Resample(orig_freq=sr, new_freq=audio.frame_rate)(denoised_vocals)

    audio_bytes = tensor_to_audio_bytes(denoised_vocals, audio.frame_rate)

    audio_seg = AudioSegment.from_wav(audio_bytes)

    return audio_seg

def tensor_to_audio_bytes(audio: torch.Tensor, sr):

    tensor = audio.cpu().numpy()
    tensor = (tensor * np.iinfo(np.int16).max).astype(np.int16)

    audio_bytes = io.BytesIO()
    torchaudio.save(audio_bytes, torch.tensor(tensor), sr, format="wav")
    audio_bytes.seek(0)

    return audio_bytes

def audiosegment_to_tensor(audio: AudioSegment, target_sr=44100):
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    samples /= np.iinfo(audio.array_type).max

    num_channels = audio.channels
    samples = samples.reshape((-1, num_channels)).T
    wav = torch.tensor(samples)

    if audio.frame_rate != target_sr:
        wav = torchaudio.transforms.Resample(orig_freq=audio.frame_rate, new_freq=target_sr)(wav)

    return wav, target_sr


if __name__ == "__main__":
    audio_path = os.path.join(os.getcwd(), "movie_audio.wav")
    rate = 16000

    # aud_buffer = extract_audio(video_path, frame_rate=rate)
    audio = AudioSegment.from_file(audio_path, format="wav")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    audio = denoise_audio(audio, device)

    non_silent_ranges = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)

    # pipe = pipeline(
    #     "automatic-speech-recognition",
    #     model="openai/whisper-large-v3",
    #     chunk_length_s=30,
    #     device=device,
    # )

    for idx, (start, end) in enumerate(non_silent_ranges):
        print(f"AUDIO INDEX {idx}")
        start = start - 500 if start - 500 > 0 else start
        end = end + 500 if (end + 500) < len(audio) else end

        audio_chunk = audio[start:end]
        # chunk_io = io.BytesIO()

        # audio_chunk.export(chunk_io, format="wav")
        # chunk_io.seek(0)

        # prediction = pipe(np.array(audio_chunk.get_array_of_samples()), batch_size=8, return_timestamps=True, return_language=True)
        # print(prediction)
