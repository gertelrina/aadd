# !pip install pydub
# sudo apt install ffmpeg  # для Linux
from pydub import AudioSegment
import os

def split_wav_to_flac_chunks(wav_path, output_dir, chunk_length_sec=10):
    # Загрузка WAV
    audio = AudioSegment.from_wav(wav_path)
    duration_ms = len(audio)
    chunk_ms = chunk_length_sec * 1000

    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    num_chunks = (duration_ms + chunk_ms - 1) // chunk_ms  # округление вверх

    for i in range(num_chunks):
        start = i * chunk_ms
        end = min((i + 1) * chunk_ms, duration_ms)
        chunk = audio[start:end]
        chunk_filename = f"{base_name}_chunk_{i:03d}.flac"
        chunk_path = os.path.join(output_dir, chunk_filename)
        chunk.export(chunk_path, format="flac")
        print(f"Saved: {chunk_path} ({(end - start) / 1000:.1f} sec)")

    print(f"\nTotal chunks: {num_chunks}")

# Пример использования
wav_file = "/home/user/agertel/dipl/Noises/NoiseX-92/babble.wav"
output_folder = "/home/user/agertel/dipl/Noises/NoiseX-92/babble/"
split_wav_to_flac_chunks(wav_file, output_folder)
