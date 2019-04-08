# -*- coding: utf-8 -*-
import csv
import numpy as np

# Ранжируем облачность на группы
dictionary_Nh = {
    'Облаков нет.': 1,
    '10%  или менее, но не 0': 1,
    '20–30%.': 1,
    '40%.': 2,
    '50%.': 2,
    '60%.': 2,
    '70 – 80%.': 3,
    '90  или более, но не 100%': 3,
    '100%.': 3,
}

# Ранжируем виды облаков на группы
dictionary_Cl = {
    'Слоисто-кучевые, образовавшиеся не из кучевых.': 1,
    'Слоистые туманообразные или слоистые разорванные, либо те и другие, но не относящиеся к облакам плохой погоды.': 2,
    'Слоистые разорванные или кучевые разорванные облака плохой погоды, либо те и другие вместе (разорванно-дождевые); обычно расположены под слоистыми или слоисто-дождевыми облаками.': 3,
    'Слоисто-кучевых, слоистых, кучевых или кучево-дождевых облаков нет.': 4,
    'Кучево-дождевые лысые с кучевыми, слоисто-кучевыми или слоистыми, либо без них.': 5,
    'Кучево-дождевые волокнистые (часто с наковальней), либо с кучево-дождевыми лысыми, кучевыми, слоистыми, разорванно-дождевыми, либо без них.': 6,
    'Кучевые плоские или кучевые разорванные, или те и другие вместе, не относящиеся к облакам плохой погоды.': 7,
    'Слоисто-кучевые, образовавшиеся из кучевых.': 8,
    'Кучевые средние или мощные или вместе с кучевыми разорванными, или с кучевыми плоскими, или со слоисто-кучевыми, либо без них; основания всех этих облаков расположены на одном уровне.': 9,
    'Кучевые и слоисто-кучевые (но не слоисто-кучевые, образовавшиеся из кучевых), основания расположены на разных уровнях.': 10
}


# Ранжируем температуру на группы
def get_group_T(value):
    value = float(value)
    if value < -5.0:
        return 1
    elif value < 10:
        return 2
    else:
        return 3


# Ранжируем влажность на группы
def get_group_U(value):
    value = int(value)
    if value < 50:
        return 1
    elif value < 80:
        return 2
    else:
        return 3


# Получаем осадки из файла. Если ячейка содержит float (уровень осадок), то записываем, что осадки были
def get_rainfall_RRR_value(RRR):
    # Получаем значение: есть осадки или нет
    dictionary_RRR = {
        'Осадков нет': 0,
        'Следы осадков': 1,  # Сюда относятся все числовые значения количества осадков
    }
    rainfall = dictionary_RRR.get(RRR)
    if rainfall is not None:
        return rainfall
    return 1


# Получаем кортеж для обучения
def get_training_tuple(Nh, Cl, T, U, RRR):
    group_Nh = dictionary_Nh.get(Nh)
    group_Cl = dictionary_Cl.get(Cl)
    group_T = get_group_T(T)
    group_U = get_group_U(U)

    values = np.array([group_Nh, group_Cl, group_T, group_U])

    rain = get_rainfall_RRR_value(RRR)
    return values, rain

def load_file():
    with open('weather.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        header = next(csv_reader)  # Пропускаем первую строку

        feats = []
        predicts = []
        for row in csv_reader:
            # Решаем проблемы с кодировкой в Python 2.7.9
            for index, item in enumerate(row):
                row[index] = row[index].decode('cp1251').encode('utf8')

            if row[17] and row[16] and row[1] and row[5] and row[23]:
                feat, predict = get_training_tuple(row[17], row[16], row[1], row[5], row[23])
                feats.append(feat)
                predicts.append(predict)
        feats = np.array(feats)
        predicts = np.array(predicts)
        return feats, predicts

def vectorized_result(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

def get_x_y_train():
    feats, predicts = load_file()
    training_inputs = [np.reshape(x, (4, 1)) for x in feats]
    training_results = [vectorized_result(y) for y in predicts]

    training_data = zip(training_inputs, training_results)

    test_inputs = [np.reshape(x, (4, 1)) for x in feats]
    test_data = zip(test_inputs, predicts)

    return training_data, test_data