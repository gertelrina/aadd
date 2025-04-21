
# ########################################################################### DATASET ##############################
import torch
import torch.nn as nn
from torchvision import models
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


class FlacMelMultilabelAugDataset(Dataset):
    def __init__(self, main_path_csv, main_path_audio, csv_path, audio_dir, sr=16000, n_mels=128, transform=None, num_classes = 8, classes= None):
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
        self.num_classes = num_classes
        self.classes = classes
        self.main_path_csv = main_path_csv
        self.main_path_audio = main_path_audio
        self.tr =  transforms.ToTensor()

    def __len__(self):
        return len(self.audio_dir)

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
        return mel_tensor_transformed, torch.tensor(label, dtype=torch.float32), file_path

classes = {
    # "None" : 0,
    "AddGaussianNoise" : 0,
    "PitchShift" : 1,
    "TimeStretch" : 2,
    "Shift" : 3,
    "ClippingDistortion" : 4,
    "LowPassFilter" : 5,
    "HighPassFilter" : 6,
    "None" : 7,
    }
class_num = len(classes)

my_transform = transforms.Compose([
    transforms.ToTensor(),
])

# main_path_audio = '/home/user/agertel/dipl/data/augmented/val_mels/imgs/'
# main_path_csv = '/home/user/agertel/dipl/data/augmented/val_mels/labels/'

main_path_audio = '/home/user/agertel/dipl/data/augmented/val_mels_separatte/imgs/'
main_path_csv = '/home/user/agertel/dipl/data/augmented/val_mels_separatte/labels/'

# main_path_audio = '/home/user/agertel/dipl/data/augmented/train_mels/imgs/'
# main_path_csv = '/home/user/agertel/dipl/data/augmented/train_mels/labels/'

val_flacs = os.listdir(main_path_audio)
val_labels = os.listdir(main_path_csv)

val_dataset = FlacMelMultilabelAugDataset(main_path_audio = main_path_audio, main_path_csv = main_path_csv, audio_dir=val_flacs, csv_path=val_labels, transform=my_transform, classes=classes)
# print(len(dataset))
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
criterion = nn.CrossEntropyLoss()
val_tqdm = tqdm(val_dataloader, desc=f"- Validate")
# # Функция вычисления точности и индексов ошибок
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

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)
# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
# num_features = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(num_features, 8)

# model = model.to(device)

# # Загрузка сохранённых весов модели
# model_path = "/home/user/agertel/dipl/efficientnet_b0_sep2_mels_2025-03-07_20-39-01.pth" #"/home/user/agertel/dipl/efficientnet_b0_sep2_mels_2025-02-13_15-53-48.pth"
# model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])

# print("Model loaded successfully!")

# model.eval()
# val_loss = 0.0
# val_corrects = 0
# val_corrects_by_one = 0

# all_preds = []
# all_labels = []


# total_zero = 0
# error_in_zero = 0
# error_not_in_zero = 0
# with torch.no_grad():
#     for images, labels, file_paths in val_tqdm:
        
#         images, labels = images.to(device), labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         val_loss += loss.item() * images.size(0)

#         # print(torch.round(torch.nn.functional.softmax(outputs, dim=-1)*1000)/1000)
        
#         # print(f'{outputs=}')
#         # print(f'{torch.round(torch.sigmoid(outputs)*1000)/1000=}')
#         preds = (torch.sigmoid(outputs) > 0.5).float()
#         # print(f'{preds=}')
#         # print(f'{labels=}')

#         for i in labels:
#             is_all_zeros = torch.all(i == 0)
#             if is_all_zeros:
#                 total_zero+=1

#         # max_indices = torch.argmax(outputs, dim=-1)
#         # preds = torch.nn.functional.one_hot(max_indices, num_classes=outputs.shape[1])

#         accuracies = calculate_accuracies(labels, preds)
#         val_corrects += accuracies['total_acc_final']
#         val_corrects_by_one += accuracies['total_acc_by_one_final']
#         error_indices = accuracies['error_indices']
#         for k in error_indices:
#             # print(f'Error in {file_paths[k]}, {preds[k]}, {labels[k]}')
#             is_all_zeros = torch.all(labels[k] == 0)
#             if is_all_zeros:
#                 error_in_zero+=1
#             else:
#                 error_not_in_zero+=1

        
#         # exit(-1)
#         all_preds.append(preds.cpu())
#         all_labels.append(labels.cpu())

#         val_tqdm.set_postfix({
#             "loss": val_loss / len(val_dataset),
#             "accuracy": val_corrects / len(val_dataset),
#             "accuracy_by_one": val_corrects_by_one / (len(val_dataset) *  class_num),
#             'total_zero': total_zero,
#             'error_in_zero': error_in_zero,
#             'error_not_in_zero' : error_not_in_zero,
#         })


# val_loss = val_loss / len(val_dataset)
# val_acc = val_corrects / len(val_dataset)
# acc_by_one = val_corrects_by_one / (len(val_dataset) *  class_num)


# # Объединяем все предсказания и метки
# all_preds = torch.cat(all_preds, dim=0).numpy()
# all_labels = torch.cat(all_labels, dim=0).numpy()
# print('total_zero', total_zero,
#             'error_in_zero', error_in_zero,
#             'error_not_in_zero', error_not_in_zero,)
# # Генерация отчёта
# from sklearn.metrics import classification_report
# report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0, target_names = ["AddGaussianNoise", "PitchShift" , "TimeStretch", "Shift", "ClippingDistortion", "LowPassFilter" , "HighPassFilter", "none" ])
# print(report)


from collections import Counter
def compute_class_statistics(dataloader):
    """
    Подсчитывает количество примеров для каждого из 7 классов.

    Args:
        dataloader (DataLoader): PyTorch DataLoader с one-hot метками.

    Returns:
        torch.Tensor: Тензор с количеством примеров для каждого класса.
    """
    total_counts = torch.zeros(8)  # Инициализация счётчика для каждого класса
    vector_counter = Counter()

    for _, labels,_ in dataloader:
        total_counts += labels.sum(dim=0)  # Суммируем 1 по каждому классу
        label_strings = ["".join(map(str, label.int().tolist())) for label in labels]
        vector_counter.update(label_strings)
        # print(total_counts,  dict(vector_counter))

    return total_counts,  dict(vector_counter)

class_statistics, vector_frequencies = compute_class_statistics(val_tqdm)

# Вычисление частоты встречаемости уникальных векторов меток
# vector_frequencies = compute_vector_frequencies(val_tqdm)

# Вывод результатов
print("Количество примеров для каждого класса:", class_statistics)
print("Частота встречаемости уникальных векторов меток:")
for vector, count in vector_frequencies.items():
    print(f"{vector}: {count}")
