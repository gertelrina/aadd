
# # # ########################################################################### DATASET ##############################
# import torch
# import torch.nn as nn
# from torchvision import models
# import os
# import csv
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
# import torchvision
# import torchvision.transforms as transforms
# import torchvision.models as models
# import librosa
# import librosa.display
# import soundfile as sf
# import matplotlib.pyplot as plt
# from io import BytesIO
# from PIL import Image
# from tqdm import tqdm
# from torch.utils.data import Dataset, DataLoader
# from audiomentations import (
#     Compose, AddGaussianNoise, PitchShift, Shift, TimeStretch,
#     ClippingDistortion, LowPassFilter, HighPassFilter, OneOf
# )

# import torch
# from torch.utils.data import Dataset
# import librosa
# import numpy as np
# import pandas as pd
# import os
# import torchvision.transforms.functional as F
# from PIL import Image
# import torchvision.transforms as transforms


# import os
# import numpy as np
# import pandas as pd
# import librosa
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count
# from PIL import Image
# import matplotlib.pyplot as plt


# def load_audio(audio_path, sample_rate=None):
#     """Загружает аудиофайл."""
#     audio, sr = librosa.load(audio_path, sr=sample_rate)
#     return audio, sr


# def extract_melspectrogram(audio, sr, n_mels=128):
#     """Преобразует аудио в мел-спектрограмму."""
#     S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
#     S_dB = librosa.power_to_db(S, ref=np.max)

#     plt.figure(figsize=(4, 5))
#     librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr)
#     plt.axis("off")

#     # Сохранение изображения спектрограммы
#     p = "val_audio.png"
#     plt.savefig(p, bbox_inches="tight", pad_inches=0)
#     plt.close()
#     return p


# def get_labels(csv_path, timestamps):
#     """Извлекает метки для каждого временного отрезка."""
#     classes = {
#     # "None" : 0,
#     "AddGaussianNoise" : 0,
#     "PitchShift" : 1,
#     "TimeStretch" : 2,
#     "Shift" : 3,
#     "ClippingDistortion" : 4,
#     "LowPassFilter" : 5,
#     "HighPassFilter" : 6,
#     "None" : 7,
#     }
#     df = pd.read_csv(csv_path)
#     labels = [[0] * len(classes) for i in range(len(timestamps))]
#     # print(labels)
#     for i, (start, end) in enumerate(timestamps):
#         row = df[(df["timestamp_start"] == start) & (df["timestamp_end"] == end)]
#         # print(row.iloc[0])
#         if pd.isna(row.iloc[0]['augmentation']) or row.iloc[0]['augmentation'] == 'None':
#                 labels[i][classes["None"]] = 1
#         else:    
#             labels[i][classes[row.iloc[0]['augmentation']]] = 1

#         # labels.append(row.iloc[0].to_dict() if not row.empty else None)
#     return labels


# def process_audio(audio_path, csv_path, transform, segment_duration=1):
#     """Обрабатывает аудиофайл, разделяя его на сегменты и преобразовывая в батч спектрограмм."""
#     audio, sr = load_audio(audio_path)
#     samples_per_segment = sr * segment_duration
#     n_segments = len(audio) // samples_per_segment
    
#     segments = []
#     timestamps = []
    
#     for i in range(n_segments):
#         segment = audio[i * samples_per_segment : (i + 1) * samples_per_segment]
#         path_img_segm = extract_melspectrogram(segment, sr)

#         mel_tensor = Image.open(path_img_segm).convert('RGB')
#         os.remove(path_img_segm)
#         img = transform(mel_tensor)
#         segments.append(img)

#         timestamps.append((i, i + 1))
    
#     labels = get_labels(csv_path, timestamps)
#     return torch.stack(segments), labels


# def process_and_return_batch(audio_path, csv_path):
#     """Главная функция: обрабатывает аудио и возвращает батч тензоров и меток."""
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#     ])
    
#     batch_tensors, labels = process_audio(audio_path, csv_path,transform)

#     return batch_tensors, torch.tensor(labels)

# def calculate_accuracies(label, pred):
#     """
#     Вычисляет общую точность и точность по отдельным элементам,
#     а также возвращает индексы, где модель ошиблась.

#     Args:
#         label (torch.Tensor): Истинные метки (тензор бинарных меток).
#         pred (torch.Tensor): Предсказанные метки (тензор бинарных меток).

#     Returns:
#         dict: Словарь с общей точностью, точностью по элементам и индексами ошибок.
#     """
#     # Вычисление общей точности (все классы в сэмпле должны быть предсказаны правильно)
#     total_acc = torch.sum((pred == label).all(dim=1)).item()

#     # Вычисление точности по отдельным элементам
#     total_acc_by_one = torch.sum(pred == label).item()

#     # Поиск индексов, где модель ошиблась
#     error_indices = torch.where((pred != label).any(dim=1))[0].tolist()

#     return {
#         "total_acc_final": total_acc,
#         "total_acc_by_one_final": total_acc_by_one,
#         "error_indices": error_indices  # Индексы ошибочных предсказаний
#     }
# batch_tensors, labels = process_and_return_batch('/home/user/agertel/dipl/data/augmented/val_multilabel_audio_all/61-70968-0000_4.flac','/home/user/agertel/dipl/data/augmented/val_multilabel_labels_all/61-70968-0000_4.csv')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# num_features = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(num_features, 8)

# model = model.to(device)

# # Загрузка сохранённых весов модели
# model_path = "/home/user/agertel/dipl/efficientnet_b0_sep2_mels_2025-03-11_12-01-06.pth" #"/home/user/agertel/dipl/efficientnet_b0_sep2_mels_2025-02-13_15-53-48.pth"
# model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])

# print("\n Model loaded successfully!")

# model.eval()

# images, labels = batch_tensors.to(device), labels.to(device)

# outputs = model(images)

# preds = (torch.sigmoid(outputs) > 0.5).float()
# acc = calculate_accuracies(labels, preds)

# print('\n Preds: \n', preds)
# print('\n Labels: \n', labels)
# print('\n Acc: \n', acc['total_acc_final'] / len(labels),  acc['total_acc_by_one_final'] / (len(labels) * 8))

            
import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.transforms.functional as TF
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from audiomentations import (
    Compose, AddGaussianNoise, PitchShift, Shift, TimeStretch,
    ClippingDistortion, LowPassFilter, HighPassFilter, OneOf
)
from multiprocessing import Pool, cpu_count


# ========================== Функции для обработки аудио ==========================

def load_audio(audio_path, sample_rate=None):
    """Загружает аудиофайл."""
    audio, sr = librosa.load(audio_path, sr=sample_rate)
    return audio, sr


def extract_melspectrogram(audio, sr, n_mels=128):
    """Преобразует аудио в мел-спектрограмму и сохраняет в виде изображения."""
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 5))
    librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr)
    plt.axis("off")

    path_img = "val_audio.png"
    plt.savefig(path_img, bbox_inches="tight", pad_inches=0)
    plt.close()
    return path_img


def get_labels(csv_path, timestamps):
    """Извлекает метки для каждого временного отрезка из CSV файла."""
    classes = {
        "AddGaussianNoise": 0,
        "PitchShift": 1,
        "TimeStretch": 2,
        "Shift": 3,
        "ClippingDistortion": 4,
        "LowPassFilter": 5,
        "HighPassFilter": 6,
        "None": 7,
    }

    df = pd.read_csv(csv_path)
    labels = [[0] * len(classes) for _ in range(len(timestamps))]

    for i, (start, end) in enumerate(timestamps):
        row = df[(df["timestamp_start"] == start) & (df["timestamp_end"] == end)]
        
        if row.empty or pd.isna(row.iloc[0]['augmentation']) or row.iloc[0]['augmentation'] == 'None':
            labels[i][classes["None"]] = 1
        else:
            labels[i][classes[row.iloc[0]['augmentation']]] = 1

    return labels


def process_audio(audio_path, csv_path, transform, segment_duration=1):
    """Разделяет аудио на сегменты, преобразует в спектрограммы и извлекает метки."""
    audio, sr = load_audio(audio_path)
    samples_per_segment = sr * segment_duration
    n_segments = len(audio) // samples_per_segment
    if n_segments == 0:
        n_segments = 1
    # print(len(audio))
    # print(n_segments)

    segments = []
    timestamps = []

    for i in range(n_segments):
        # print('hello')
        segment = audio[i * samples_per_segment: (i + 1) * samples_per_segment]
        path_img = extract_melspectrogram(segment, sr)

        mel_tensor = Image.open(path_img).convert('RGB')
        os.remove(path_img)
        img = transform(mel_tensor)
        segments.append(img)

        timestamps.append((i, i + 1))
    labels = None
    if csv_path is not None:
        labels = get_labels(csv_path, timestamps)
    return torch.stack(segments), labels


def process_and_return_batch(audio_path, csv_path):
    """Обрабатывает аудиофайл и возвращает батч спектрограмм и меток."""
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    batch_tensors, labels = process_audio(audio_path, csv_path, transform)
    if labels is not None:
        return batch_tensors, torch.tensor(labels)
    else: 
        return batch_tensors, None


# ========================== Функция вычисления точности ==========================

def calculate_accuracies(label, pred):
    """
    Вычисляет общую точность, точность по элементам и индексы ошибок.

    Args:
        label (torch.Tensor): Истинные бинарные метки.
        pred (torch.Tensor): Предсказанные бинарные метки.

    Returns:
        dict: Словарь с точностью и индексами ошибок.
    """
    total_acc = torch.sum((pred == label).all(dim=1)).item()
    total_acc_by_one = torch.sum(pred == label).item()
    error_indices = torch.where((pred != label).any(dim=1))[0].tolist()

    return {
        "total_acc_final": total_acc,
        "total_acc_by_one_final": total_acc_by_one,
        "error_indices": error_indices
    }


# # ========================== Загрузка данных и модели ==========================

# audio_path = "/home/user/agertel/dipl/data/augmented/augmented_total_audio_and_labels/val_multilabel_audio_all/61-70968-0000_1.flac"

# csv_path = "/home/user/agertel/dipl/data/augmented/augmented_total_audio_and_labels/val_multilabel_labels_all/61-70968-0000_0.csv"

# batch_tensors, labels = process_and_return_batch(audio_path, csv_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# # Загрузка предобученной модели
# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# num_features = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(num_features, 8)
# model = model.to(device)

# # Загрузка сохранённых весов
# model_path = "/home/user/agertel/dipl/efficientnet_b0_sep2_mels_2025-03-11_12-01-06.pth"
# model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
# print("\nModel loaded successfully!")

# model.eval()

# # ========================== Предсказание и оценка точности ==========================

# images, labels = batch_tensors.to(device), labels.to(device)
# outputs = model(images)
# preds = (torch.sigmoid(outputs) > 0.5).float()
# preds2 = torch.round((torch.sigmoid(outputs) * 1000)) / 1000

# acc = calculate_accuracies(labels, preds)

# # Вывод результатов
# print("\nPredictions:\n", preds2)
# print("\nLabels:\n", labels)
# print("\nAccuracy:")
# print("Total Accuracy:", acc["total_acc_final"] / len(labels))
# print("Per Label Accuracy:", acc["total_acc_by_one_final"] / (len(labels) * 8))



# ========================== Загрузка данных и модели ==========================

# import torch
# import torch.nn as nn
# import os
import glob
from collections import defaultdict
# from torchvision import models
# from process import process_and_return_batch  # замените на свою функцию
# from metrics import calculate_accuracies      # замените на свою функцию

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Классы
classes = {
    0: "AddGaussianNoise",
    1: "PitchShift",
    2: "TimeStretch",
    3: "Shift",
    4: "ClippingDistortion",
    5: "LowPassFilter",
    6: "HighPassFilter",
    7: "None",
}
idx_to_class = {v: k for k, v in classes.items()}

# Порог вероятности
THRESHOLD = 0.001  # выводить только классы с вероятностью выше этого значения

# Счётчики
clean_count = 0
corrupted_count = 0
corruption_per_class = defaultdict(int)  # {class_name: count}

# Загрузка модели
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 8)
model = model.to(device)

# Загрузка весов
model_path = "/home/user/agertel/dipl/efficientnet_b0_sep2_mels_2025-03-11_12-01-06.pth"
checkpoint = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
print("\nModel loaded successfully!")

model.eval()

# Путь к папке или список файлов
audio_folder = "/home/user/agertel/dipl/data/asr_flac_public_phone"
audio_files = glob.glob(os.path.join(audio_folder, "**", "*.flac"), recursive=True)
# audio_files = ['/home/user/agertel/dipl/data/output.flac',]
# '/home/user/agertel/dipl/data/LibriSpeech/dev-other/116/288045/116-288045-0016.flac',
# '/home/user/agertel/dipl/data/LibriSpeech/dev-other/116/288048/116-288048-0006.flac']

for audio_path in audio_files:
    # print(f"\n==================== {os.path.basename(audio_path)} ====================")
    print(f"\n==================== {audio_path} ====================")
    # print(audio_path)

    csv_path = None
    batch_tensors, labels = process_and_return_batch(audio_path, csv_path)

    if labels is not None:
        images, labels = batch_tensors.to(device), labels.to(device)
    else:
        images = batch_tensors.to(device)

    outputs = model(images)
    probs = torch.sigmoid(outputs)
    preds = (probs > 0.5).float()
    preds_rounded = torch.round(probs * 1000) / 1000  # округление для вывода

    file_has_corruption = False
    classes_found_in_file = set()

    # Логгирование по секундам и классам с учетом трешхолда
    for i, second_probs in enumerate(preds_rounded):
        relevant = [(classes[cls_idx], prob.item()) for cls_idx, prob in enumerate(second_probs) if prob.item() >= THRESHOLD]
        non_none = [(name, p) for name, p in relevant if name != "None"]
        if non_none:
            file_has_corruption = True
            print(f"\n--- Second {i:02d} ---")
            for class_name, prob in non_none:
                print(f"{class_name:<20} | Prob: {prob:.3f}")
                classes_found_in_file.add(class_name)

    # Обновляем счётчики
    if file_has_corruption:
        corrupted_count += 1
        for class_name in classes_found_in_file:
            corruption_per_class[class_name] += 1
    else:
        clean_count += 1

    if labels is not None:
        acc = calculate_accuracies(labels, preds)
        print(f"\nAccuracy for {os.path.basename(audio_path)}:")
        print(f"  Total Accuracy: {acc['total_acc_final'] / len(labels):.3f}")
        print(f"  Per Label Accuracy: {acc['total_acc_by_one_final'] / (len(labels) * 8):.3f}")

# Финальная статистика
print("\n================ SUMMARY ================\n")
print(audio_folder.split('/')[-1])
print(f"Total audio files:     {len(audio_files)}")
print(f"Clean recordings:      {clean_count}")
print(f"Corrupted recordings:  {corrupted_count}")
print("\nCorruption frequency by class:")
for class_name in sorted(corruption_per_class.keys()):
    print(f"{class_name:<20} : {corruption_per_class[class_name]}")
