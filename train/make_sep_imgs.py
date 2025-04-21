# import os
# import csv
# import numpy as np
# import pandas as pd
# # import torch
# # import torch.nn as nn
# # import torch.optim as optim
# # import torch.optim.lr_scheduler as lr_scheduler
# # import torchvision
# # import torchvision.transforms as transforms
# # import torchvision.models as models
# import librosa
# import librosa.display
# import soundfile as sf
# import matplotlib.pyplot as plt
# from io import BytesIO
# from PIL import Image
# from tqdm import tqdm


# def get_row_by_timestamps(csv_data, timestamp_start: int, timestamp_end: int):
#     """Извлекает строку из CSV по временным меткам."""
    

#     # Фильтруем строку по заданным значениям
#     row = df[(df["timestamp_start"] == timestamp_start) & (df["timestamp_end"] == timestamp_end)]

#     return row.iloc[0].to_dict() if not row.empty else None


# def process_audio(input_audio_path: str, csv_path: str, name: str, save_dir: str):
#     """
#     Разбивает аудио на 1-секундные части, сохраняет спектрограммы и метаданные.

#     :param input_audio_path: str, путь к входному аудио-файлу
#     :param csv_path: str, путь к CSV с метаданными
#     :param name: str, имя выходных файлов
#     :param save_dir: str, директория для сохранения результатов
#     """
#     save_dir_imgs = os.path.join(save_dir, "imgs")
#     save_dir_labels = os.path.join(save_dir, "labels")
#     os.makedirs(save_dir_imgs, exist_ok=True)
#     os.makedirs(save_dir_labels, exist_ok=True)

#     # Загружаем аудио
#     audio, sample_rate = librosa.load(input_audio_path, sr=None)
#     samples_per_second = sample_rate
#     n_parts = len(audio) // samples_per_second

#     if n_parts == 0:
#         print(f"⚠️ Аудио слишком короткое: {input_audio_path}")
#         return

#     csv_data = pd.read_csv(csv_path)

#     for i in range(n_parts):
#         part = audio[i * samples_per_second : (i + 1) * samples_per_second]

#         # Читаем метаданные
#         csv_out_path = os.path.join(save_dir_labels, f"{name}_{i}.csv")
#         row_data = get_row_by_timestamps(csv_data, i, i + 1)

#         if row_data:
#             with open(csv_out_path, mode="w", newline="") as csv_file:
#                 writer = csv.DictWriter(csv_file, fieldnames=row_data.keys())
#                 writer.writeheader()
#                 writer.writerow(row_data)

#         # Создание мел-спектрограммы
#         S = librosa.feature.melspectrogram(y=part, sr=sample_rate, n_mels=128)
#         S_dB = librosa.power_to_db(S, ref=np.max)

#         plt.figure(figsize=(4, 5))
#         librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sample_rate)
#         plt.axis("off")

#         # Сохранение изображения спектрограммы
#         plt.savefig(os.path.join(save_dir_imgs, f"{name}_{i}.png"), bbox_inches="tight", pad_inches=0)
#         plt.close()


# # Пути к данным
# p_audio = "/home/user/agertel/dipl/train_multilabel_audio_all/"
# p_label = "/home/user/agertel/dipl/train_multilabel_labels_all/"
# save_dir = "/home/user/agertel/dipl/separate_train/"
# os.makedirs(save_dir, exist_ok=True)

# # Обработка всех аудиофайлов
# for x in tqdm(os.listdir(p_audio), desc="Обработка аудио"):
#     name = os.path.splitext(x)[0]
#     csv_p = os.path.join(p_label, f"{name}.csv")
#     audio_p = os.path.join(p_audio, x)
#     process_audio(audio_p, csv_p, name, save_dir)

import os
import csv
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


plt.switch_backend("Agg")  # Отключаем GUI backend, ускоряя сохранение изображений


def get_row_by_timestamps(df, timestamp_start: int, timestamp_end: int):
    """Извлекает строку из CSV по временным меткам."""
    row = df[(df["timestamp_start"] == timestamp_start) & (df["timestamp_end"] == timestamp_end)]
    return row.iloc[0].to_dict() if not row.empty else None


def process_audio(params):
    """Функция для обработки одного аудиофайла, используется в пуле процессов."""
    input_audio_path, csv_path, name, save_dir = params
    save_dir_imgs = os.path.join(save_dir, "imgs")
    save_dir_labels = os.path.join(save_dir, "labels")
    os.makedirs(save_dir_imgs, exist_ok=True)
    os.makedirs(save_dir_labels, exist_ok=True)

    try:
        # Загружаем аудио
        audio, sample_rate = librosa.load(input_audio_path, sr=None)
        samples_per_second = sample_rate
        n_parts = len(audio) // samples_per_second

        if n_parts == 0:
            print(f"⚠️ Аудио слишком короткое: {input_audio_path}")
            return

        # Загружаем CSV только один раз
        csv_data = pd.read_csv(csv_path)

        for i in range(n_parts):
            part = audio[i * samples_per_second : (i + 1) * samples_per_second]

            # Читаем метаданные
            csv_out_path = os.path.join(save_dir_labels, f"{name}_{i}.csv")
            row_data = get_row_by_timestamps(csv_data, i, i + 1)

            if row_data:
                with open(csv_out_path, mode="w", newline="") as csv_file:
                    writer = csv.DictWriter(csv_file, fieldnames=row_data.keys())
                    writer.writeheader()
                    writer.writerow(row_data)

            # Создание мел-спектрограммы
            S = librosa.feature.melspectrogram(y=part, sr=sample_rate, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)

            plt.figure(figsize=(4, 5))
            librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sample_rate)
            plt.axis("off")

            # Сохранение изображения спектрограммы
            plt.savefig(os.path.join(save_dir_imgs, f"{name}_{i}.png"), bbox_inches="tight", pad_inches=0)
            plt.close()
    except Exception as e:
        print(f"Ошибка обработки файла {input_audio_path}: {e}")


def main():
    """Основная функция для параллельной обработки всех аудиофайлов."""
    p_audio = "/home/user/agertel/dipl/data/augmented/train_multilabel_audio_all/"
    p_label = "/home/user/agertel/dipl/data/augmented/train_multilabel_labels_all/"
    save_dir = "/home/user/agertel/dipl/separate_val/"
    os.makedirs(save_dir, exist_ok=True)

    audio_files = [(os.path.join(p_audio, x), os.path.join(p_label, f"{os.path.splitext(x)[0]}.csv"), 
                    os.path.splitext(x)[0], save_dir) for x in os.listdir(p_audio)]

    # Определяем количество потоков
    num_workers = min(cpu_count(), len(audio_files))
    print(f"Используется {num_workers} процессов...")

    # Параллельная обработка
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_audio, audio_files), total=len(audio_files), desc="Обработка аудио"))


if __name__ == "__main__":
    main()
