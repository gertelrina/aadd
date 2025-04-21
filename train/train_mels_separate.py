# ! pip install colorama
# ! pip install soundfile
# ! pip install gdown
# ! pip install audiomentations
# ! pip install wandb
# ! wandb login

import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
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

import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
import os
import torchvision.transforms.functional as F
from PIL import Image
import torchvision.transforms as transforms


# # Формат row:
# import os
# list_audio = []
# p = 'data/LibriSpeech/dev-clean/'
# for i in os.listdir(p):
#   if 'untitled' in i:
#     continue
#   for j in  os.listdir(p+i):
#     for k in  os.listdir(p+i+'/'+j):
#       if 'txt' in k:
#         continue
#       list_audio.append(p+i+'/'+j+'/'+k)
# import random
# # list_audio_100 = random.choices(list_audio, k=100)
# list_audio_all = list_audio
# # print(list_audio_100)  # Список из 100 элементов, возможно, с повторениями

# import os
# list_audio_val = []
# p = 'data/LibriSpeech/test-clean/'
# for i in os.listdir(p):
#   if 'untitled' in i:
#     continue
#   for j in  os.listdir(p+i):
#     for k in  os.listdir(p+i+'/'+j):
#       if 'txt' in k:
#         continue
#       list_audio_val.append(p+i+'/'+j+'/'+k)

# import random
# # list_audio_100_val = random.choices(list_audio_val, k=100)
# list_audio_all_val = list_audio_val
# # print(list_audio_100_val)

# def process_audio(input_audio_path, output_audio_path, csv_path, apply_augmentation=True):
#     """
#     Разбивает аудио на части по 1 секунде, применяет случайную аугментацию (если указано),
#     записывает метаданные в CSV и сохраняет итоговое аудио.

#     :param input_audio_path: str, путь к входному аудио-файлу
#     :param output_audio_path: str, путь для сохранения обработанного аудио
#     :param csv_path: str, путь для сохранения CSV с метаданными
#     :param apply_augmentation: bool, применять ли аугментацию
#     """
#     # Load the audio
#     audio, sample_rate = librosa.load(input_audio_path, sr=None)

#     # Calculate the number of samples in 1 second
#     samples_per_second = sample_rate
#     n_parts = len(audio) // samples_per_second

#     # Split audio into 1-second parts
#     audio_parts = [
#         audio[i * samples_per_second:(i + 1) * samples_per_second]
#         for i in range(n_parts)
#     ]

#     # List to store augmented audio parts
#     processed_parts = []
#     csv_data = []
#     print(len(audio_parts))
#     if len(audio_parts) == 0:
#       print('!!')
#       return
#     for i, part in enumerate(audio_parts):
#         if apply_augmentation:
#             apply_flag = False

#             # Define augmentation
#             hard_augment = OneOf([
#                 AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=1.0),
#                 PitchShift(min_semitones=-8, max_semitones=8, p=1.0),
#                 TimeStretch(min_rate=0.5, max_rate=1.5, p=1.0),
#                 Shift(min_shift=-1.0, max_shift=1.0, p=1.0),
#                 ClippingDistortion(min_percentile_threshold=0, max_percentile_threshold=20, p=1.0),
#                 LowPassFilter(min_cutoff_freq=150, max_cutoff_freq=750, p=1.0),
#                 HighPassFilter(min_cutoff_freq=2000, max_cutoff_freq=8000, p=1.0)
#             ], p=0.5)

#             # Apply augmentation
#             processed_part = hard_augment(samples=part, sample_rate=sample_rate)

#             applied_augmentation = None
#             parameters = None

#             for transform in hard_augment.transforms:
#                 if transform.parameters.get('should_apply') is not None:
#                     applied_augmentation = transform.__class__.__name__
#                     parameters = transform.parameters
#                     apply_flag = True
#                     break

#             if not apply_flag:
#                 applied_augmentation = 'None'
#                 parameters = ''
#         else:
#             processed_part = part
#             applied_augmentation = 'None'
#             parameters = ''

#         processed_parts.append(processed_part)

#         # Calculate timestamps
#         start_time = i  # Start time in seconds
#         end_time = i + 1  # End time in seconds

#         # Add to CSV data
#         csv_data.append({
#             "timestamp_start": start_time,
#             "timestamp_end": end_time,
#             "augmentation": applied_augmentation,
#             "parameters": parameters,
#             "sample_rate": sample_rate
#         })

#     # Save metadata to CSV
#     with open(csv_path, mode='w', newline='') as csv_file:
#         fieldnames = ["timestamp_start", "timestamp_end", "augmentation", "parameters", "sample_rate"]
#         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(csv_data)

#     # Combine processed parts
#     final_audio = np.concatenate(processed_parts)

#     # Save the processed audio
#     sf.write(output_audio_path, final_audio, sample_rate)


# def process_audio_list(audio_list, n, output_dir, csv_dir):
#     """
#     Обрабатывает список аудиофайлов, применяя N раз аугментацию к каждому,
#     и сохраняет результаты, включая оригинальное аудио без аугментации.

#     :param audio_list: list, список путей к аудиофайлам
#     :param n: int, количество раз для обработки каждого файла
#     :param output_dir: str, папка для сохранения обработанных аудио
#     :param csv_dir: str, папка для сохранения CSV-файлов
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(csv_dir, exist_ok=True)
#     N = 0
#     for audio_path in tqdm(audio_list):
#         N+=1

#         base_name = os.path.splitext(os.path.basename(audio_path))[0]

#         # Сохранение оригинального аудио без аугментации
#         output_audio_path = os.path.join(output_dir, f"{base_name}_0.flac")
#         csv_path = os.path.join(csv_dir, f"{base_name}_0.csv")

#         process_audio(
#             input_audio_path=audio_path,
#             output_audio_path=output_audio_path,
#             csv_path=csv_path,
#             apply_augmentation=False
#         )

#         # Применение аугментаций
#         M = 0
#         print(output_audio_path)
#         for i in range(n):
#             M+=1
#             output_audio_path = os.path.join(output_dir, f"{base_name}_{i+1}.flac")
#             csv_path = os.path.join(csv_dir, f"{base_name}_{i+1}.csv")
#             try:
#               process_audio(
#                 input_audio_path=audio_path,
#                 output_audio_path=output_audio_path,
#                 csv_path=csv_path
#             )
#             except Exception as e:
#               print(e)
#               print(i)
#               print(audio_path)

# # Пример использования
# process_audio_list(
#     audio_list=list_audio_all,
#     n=10,
#     output_dir="train_multilabel_audio_all",
#     csv_dir="train_multilabel_labels_all"
# )


# # Пример использования
# process_audio_list(
#     audio_list=list_audio_all_val,
#     n=10,
#     output_dir="val_multilabel_audio_all",
#     csv_dir="val_multilabel_labels_all"
# )

# def save_mel_spectrogram(audio_path: str, n_mels: int, output_path: str = 'spectrogram.png') -> torch.Tensor:
#     """
#     Генерирует мел-спектрограмму аудиофайла и возвращает её в виде тензора.

#     Parameters:
#     - audio_path (str): путь к аудиофайлу
#     - n_mels (int): количество мел-банков
#     - output_path (str): путь для сохранения изображения (по умолчанию 'spectrogram.png')

#     Returns:
#     - torch.Tensor: мел-спектрограмма в формате тензора (C, H, W)
#     """
#     # Загрузка аудиофайла
#     y, sr = librosa.load(audio_path, sr=None)

#     # Вычисление мел-спектрограммы
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
#     S_dB = librosa.power_to_db(S, ref=np.max)

#     # Визуализация спектрограммы
#     plt.figure(figsize=(10, 4))
#     plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#     librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
#     plt.axis('off')

#     # Сохранение изображения в память
#     buf = BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
#     plt.close()
#     buf.seek(0)

#     # Конвертация изображения в тензор
#     img = Image.open(buf).convert('RGB')
#     tensor = torch.from_numpy(np.array(img)).float() / 255.0  # Нормализация
#     tensor = tensor.permute(2, 0, 1)  # Преобразование в формат (C, H, W)

#     return tensor

# ########################################################################### DATASET ##############################

class FlacMelMultilabelAugDataset(Dataset):
    def __init__(self, main_path_csv, main_path_audio, csv_path, audio_dir, sr=16000, n_mels=128, transform=None,classes= None):
        """
        csv_path: путь к CSV-файлу с временными диапазонами и аугментациями
        audio_dir: директория с аудиофайлами
        sr: частота дискретизации
        n_mels: количество мел-банков
        transform: трансформации torchvision (например, Resize, ToTensor и т.д.)
        """
        # self.data = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.labels_audio = csv_path #pd.read_csv(csv_path)
        self.sr = sr
        self.n_mels = n_mels
        self.transform = transform
        self.num_classes = len(classes) 
        self.classes = classes
        self.main_path_csv = main_path_csv
        self.main_path_audio = main_path_audio
        self.tr =  transforms.ToTensor()

    def __len__(self):
        return len(self.audio_dir)

    # def __getitem__(self, idx):
    #     # Получаем имя аудиофайла
    #     # filenames = self.data['filename'].unique()
    #     # filename = filenames[idx]
    #     # file_path = os.path.join(self.audio_dir, filename)

    #     file_path =  self.main_path_audio + self.audio_dir[idx]
    #     label_file =  pd.read_csv( self.main_path_csv + self.audio_dir[idx].split('.')[0] + '.csv')

    #     # Создаем метку для мультилейбл классификации
    #     label = [0] *  self.num_classes # Предположим, 8 классов

    #     # Применяем аугментации к временным диапазонам
    #     for _, row in label_file.iterrows():
    #         if pd.isna(row['augmentation']) or row['augmentation'] == 'None':
    #             continue
    #         label[classes[row['augmentation']]] = 1

    #     mel_tensor = Image.open(file_path).convert('RGB')
    #     mel_tensor_transformed = self.tr(mel_tensor)
    #     return mel_tensor_transformed, torch.tensor(label, dtype=torch.float32)

    def __getitem__(self, idx):
        # Получаем имя аудиофайла
        # filenames = self.data['filename'].unique()
        # filename = filenames[idx]
        # file_path = os.path.join(self.audio_dir, filename)

        file_path =  self.main_path_audio + self.audio_dir[idx]
        label_file =  pd.read_csv( self.main_path_csv + self.audio_dir[idx].split('.')[0] + '.csv')

        # Создаем метку для мультилейбл классификации
        label = [0] *  self.num_classes # Предположим, 8 классов

        # Применяем аугментации к временным диапазонам
        for _, row in label_file.iterrows():
            if pd.isna(row['augmentation']) or row['augmentation'] == 'None':
                label[classes["None"]] = 1
                continue
            label[classes[row['augmentation']]] = 1

        mel_tensor = Image.open(file_path).convert('RGB')
        mel_tensor_transformed = self.tr(mel_tensor)
        return mel_tensor_transformed, torch.tensor(label, dtype=torch.float32)

classes = {
    "AddGaussianNoise" : 0,
    "PitchShift" : 1,
    "TimeStretch" : 2,
    "Shift" : 3,
    "ClippingDistortion" : 4,
    "LowPassFilter" : 5,
    "HighPassFilter" : 6,
    "None" : 7,
    }

# # import torch
# # import torchvision
# # import matplotlib.pyplot as plt

# # def imshow(img, mean=None, std=None):
# #     """
# #     Вспомогательная функция для отображения изображения.
# #     При необходимости выполняет «разнормализацию» (unnormalization), 
# #     если изображения были нормализованы.
# #     """
# #     # Если заданы mean и std, то выполним обратное преобразование
# #     if mean is not None and std is not None:
# #         for i in range(img.size(0)):
# #             img[i] = img[i] * std[i] + mean[i]

# #     # Перекладываем каналы [C, H, W] -> [H, W, C]
# #     img = img.permute(1, 2, 0)
# #     # Ограничим значения в диапазоне [0, 1] на случай, если есть артефакты округления
# #     img = torch.clamp(img, 0, 1)
# #     plt.imshow(img)
# #     plt.axis('off')  # убираем оси для наглядности






# ############################################################# TRAIN ################
import torchvision.transforms as transforms
# Импортируем необходимые библиотеки
import wandb
from colorama import Fore, Style
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm

my_transform = transforms.Compose([
    transforms.ToTensor(),
])

# main_path_audio = '/home/user/agertel/dipl/data/augmented/train_mels/imgs_balance_/'
# main_path_csv = '/home/user/agertel/dipl/data/augmented/train_mels/labels_balance_/'
main_path_audio = '/home/user/agertel/dipl/data/augmented/train_mels_separatte/imgs/'
main_path_csv = '/home/user/agertel/dipl/data/augmented/train_mels_separatte/labels/'

train_flacs = os.listdir(main_path_audio)
train_labels = os.listdir(main_path_csv)
train_dataset = FlacMelMultilabelAugDataset(main_path_audio = main_path_audio, main_path_csv = main_path_csv, audio_dir=train_flacs, csv_path=train_labels, transform=my_transform, classes=classes)
# print(len(dataset))
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

for batch_mels, batch_labels in train_dataloader:
    # print(batch_mels.shape, batch_labels)
    break

my_transform = transforms.Compose([
    transforms.ToTensor(),
])

# main_path_audio = '/home/user/agertel/dipl/data/augmented/val_mels/imgs_balance_/'
# main_path_csv = '/home/user/agertel/dipl/data/augmented/val_mels/labels_balance_/'

main_path_audio = '/home/user/agertel/dipl/data/augmented/val_mels_separatte/imgs/'
main_path_csv = '/home/user/agertel/dipl/data/augmented/val_mels_separatte/labels/'

val_flacs = os.listdir(main_path_audio)
val_labels = os.listdir(main_path_csv)

val_dataset = FlacMelMultilabelAugDataset(main_path_audio = main_path_audio, main_path_csv = main_path_csv, audio_dir=val_flacs, csv_path=val_labels, transform=my_transform, classes=classes)
# print(len(dataset))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)

for batch_mels, batch_labels in val_dataloader:
    # print(batch_mels.shape, batch_labels)
    break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
num_features = model.classifier[1].in_features
num_cls = len(classes)
model.classifier[1] = nn.Linear(num_features, num_cls)

model = model.to(device)

def calculate_accuracies(label, pred):
    """
    Calculates total accuracy and total accuracy by individual elements.

    Args:
        label (torch.Tensor): Ground truth labels (binary tensor).
        pred (torch.Tensor): Predicted labels (binary tensor).

    Returns:
        dict: Dictionary containing total accuracy and accuracy by individual elements.
    """
    # Calculate total accuracy (all classes correct in a sample)
    total_acc = torch.sum((pred == label).all(dim=1)).item()
    # total_acc_final = total_acc / len(pred)

    # Calculate accuracy by individual elements
    total_acc_by_one = torch.sum(pred == label).item()
    # total_acc_by_one_final = total_acc_by_one / (len(pred) * label.size(1))

    return {
        "total_acc_final": total_acc,
        "total_acc_by_one_final": total_acc_by_one
    }

# Инициализация W&B
wandb.init(project="audio-degr-detection",
           notes="first experiment",
           tags=["separate_imgs"],
           name="bceloss_total_dataset_final")

# Инициализация параметров
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
epochs = 10
train_batch_size = train_dataloader.batch_size
val_batch_size = val_dataloader.batch_size

# Сохранение параметров в W&B
wandb.config.update({
    "train_batch_size": train_batch_size,
    "val_batch_size": val_batch_size
})

best_val_acc = 0
import datetime
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_filename = f'efficientnet_b0_sep2_mels_{timestamp}.pth'

for epoch in range(epochs):
    print(f'{epoch=}')

    # --- TRAIN ---
    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_corrects_by_one = 0
    train_tqdm = tqdm(train_dataloader, desc=f"Epoch {epoch+1} - Train")
    for images, labels in train_tqdm:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        # print(outputs, )
        # print(labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()


        # import ipdb; ipdb.set_trace()
        # Получение индексов максимальных значений после softmax
        # max_indices = torch.argmax(outputs, dim=-1)

        # # Преобразование в one-hot представление
        # preds = torch.nn.functional.one_hot(max_indices, num_classes=outputs.shape[1])


        accuracies = calculate_accuracies(labels, preds)
        running_corrects += accuracies['total_acc_final']
        running_corrects_by_one += accuracies['total_acc_by_one_final']


        train_tqdm.set_postfix({
            "loss": running_loss / len(train_dataset),
            "accuracy": running_corrects / len(train_dataset),
             "accuracy_by_one": running_corrects_by_one / (len(train_dataset) *  num_cls),

        })

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects / len(train_dataset)
    epoch_acc_by_one = running_corrects_by_one / (len(train_dataset) *  num_cls)

    wandb.log({
        "Train Loss": epoch_loss,
        "Train Accuracy": epoch_acc,
         "Train Accuracy by one": epoch_acc_by_one,

        "Learning Rate": scheduler.get_last_lr()[0],
        "Epoch": epoch
    })

    # --- VALIDATE ---
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_corrects_by_one = 0

    all_preds = []
    all_labels = []

    val_tqdm = tqdm(val_dataloader, desc=f"Epoch {epoch+1} - Validate")
    with torch.no_grad():
        for images, labels in val_tqdm:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            # val_corrects += torch.sum((preds == labels).all(dim=1)).item()
            accuracies = calculate_accuracies(labels, preds)
            val_corrects += accuracies['total_acc_final']
            val_corrects_by_one += accuracies['total_acc_by_one_final']

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            val_tqdm.set_postfix({
                "loss": val_loss / len(val_dataset),
                "accuracy": val_corrects / len(val_dataset),
                "accuracy_by_one": val_corrects_by_one / (len(val_dataset) *  num_cls),

            })


    val_loss = val_loss / len(val_dataset)
    val_acc = val_corrects / len(val_dataset)
    acc_by_one = val_corrects_by_one / (len(val_dataset) *  num_cls)


    # Объединяем все предсказания и метки
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    # Генерация отчёта
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0, target_names = list(classes.keys()))
    print(report)
    # Логирование метрик для каждого класса
    for class_label, metrics in report.items():
        if class_label.isdigit():
            wandb.log({
                f"Class {class_label} Precision": metrics["precision"],
                f"Class {class_label} Recall": metrics["recall"],
                f"Class {class_label} F1-Score": metrics["f1-score"],
                f"Class {class_label} Support": metrics["support"]
            })

    # Логирование усреднённых метрик
    wandb.log({
        "Validation Loss": val_loss,
        "Validation Accuracy": val_acc,
        "Validation Accuracy by one": acc_by_one,

        "Macro Avg Precision": report["macro avg"]["precision"],
        "Macro Avg Recall": report["macro avg"]["recall"],
        "Macro Avg F1-Score": report["macro avg"]["f1-score"],
        "Weighted Avg Precision": report["weighted avg"]["precision"],
        "Weighted Avg Recall": report["weighted avg"]["recall"],
        "Weighted Avg F1-Score": report["weighted avg"]["f1-score"]
    })

    if acc_by_one > best_val_acc:
        best_val_acc = acc_by_one
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'val_accuracy_by_one': acc_by_one
        }, model_filename)
        print(Fore.GREEN + f"Best model saved at epoch {epoch+1} with val_accuracy_by_one: {acc_by_one:.4f}" + Style.RESET_ALL)

    scheduler.step()

wandb.finish()
