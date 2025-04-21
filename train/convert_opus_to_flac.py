import os
import subprocess

source_root = "/home/user/agertel/dipl/data/asr_public_phone_calls_2"   # замени на актуальный путь

# Путь к новой папке, куда сохраняем .flac
target_root = "/home/user/agertel/dipl/data/asr_flac_public_phone"

for dirpath, _, filenames in os.walk(source_root):
    for filename in filenames:
        if filename.endswith(".opus"):
            # Полный путь к .opus файлу
            input_path = os.path.join(dirpath, filename)

            # Относительный путь от исходной папки
            relative_path = os.path.relpath(dirpath, source_root)

            # Папка назначения с сохранением структуры
            target_dir = os.path.join(target_root, relative_path)
            os.makedirs(target_dir, exist_ok=True)

            # Путь к выходному .flac файлу
            output_filename = os.path.splitext(filename)[0] + ".flac"
            output_path = os.path.join(target_dir, output_filename)

            # Команда ffmpeg
            cmd = ["ffmpeg", "-y", "-i", input_path, output_path]
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"✔ Converted: {input_path} -> {output_path}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error converting {input_path}: {e}")