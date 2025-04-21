import librosa
import librosa.display
# import matplotlib.pyplot as plt
import numpy as np

def save_mel_spectrogram(audio_path, n_mels, output_path='spectrogram.png'):
    import matplotlib.pyplot as plt
    """
    Сохраняет мел-спектрограмму аудиофайла в виде изображения.

    Parameters:
    - audio_path (str): путь к аудиофайлу
    - n_mels (int): количество мел-банков
    - output_path (str): путь для сохранения изображения (по умолчанию 'spectrogram.png')
    """
    # print(f"Загружаем аудиофайл: {audio_path}")
    y, sr = librosa.load(audio_path, sr=None)  # Загружаем аудиофайл
    # print(f"Частота дискретизации аудио: {sr} Гц")
    
    # Выводим длительность аудио
    duration = librosa.get_duration(y=y, sr=sr)
    # print(f"Длительность аудиофайла: {duration:.2f} секунд")

    # print(f"Вычисляем мел-спектрограмму... n_mels={n_mels}")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)  # Вычисляем мел-спектрограмму
    S_dB = librosa.power_to_db(S, ref=np.max)  # Переводим в децибелы

    # print(f"Размер мел-спектрограммы (в децибелах): {S_dB.shape}")

    # Строим спектрограмму
    # print("Создаем график спектрограммы...")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr)
    plt.axis('off')  # Отключаем оси

    # print(f"Сохраняем спектрограмму в файл: {output_path}")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    # print(f"Спектрограмма успешно сохранена как {output_path}")

import os
import tqdm

# p = '/home/user/agertel/dipl/train_multilabel_audio_all/'
# audios = os.listdir(p)
# for x in tqdm.tqdm(audios):
#     name = x.split('.')[0]
#     save_path = '/home/user/agertel/dipl/train_multilabel_imgs_all_total/'+name+'.png'
#     save_mel_spectrogram(p+x, 128, save_path)


p = '/home/user/agertel/dipl/data/augmented/val_multilabel_audio_all/'
audios = os.listdir(p)
for x in tqdm.tqdm(audios):
    name = x.split('.')[0]
    save_path = '/home/user/agertel/dipl/data/augmented/val_multilabel_imgs_all_total/'+name+'.png'
    save_mel_spectrogram(p+x, 128, save_path)

