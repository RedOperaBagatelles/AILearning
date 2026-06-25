# 이미지 분류하는 데이터 셋
# 이미지 크기 : 32 x 32, 클래스 수 : 10, 학습 데이터 수 : 50,000, 테스트 데이터 수 : 10,000
# 클래스 : airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

x_train = x_train / 255.0
x_test = x_test / 255.0

# One-hot encoding (ex : 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], ...)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Data Augmentation : 살짝 기울어진 고양이, 뒤집힌 고양이처럼 다양한 형태의 이미지를 만들어서 모델이 다양한 형태의 이미지를 학습할 수 있도록 함
# rotation_range : 이미지를 무작위로 회전하는 범위 (0~180)
# width_shift_range : 이미지를 무작위로 수평 이동하는 범위 (0~1, 0.1은 10% 이동)
# height_shift_range : 이미지를 무작위로 수직 이동하는 범위 (0~1, 0.1은 10% 이동)
# horizontal_flip : 이미지를 무작위로 수평으로 뒤집을지 여부
datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(x_train)

model = Sequential([
    Input(shape=(32, 32, 3)),
    
    Conv2D(32, (3, 3), activation='relu'),
    # 출력 값이 [-0.1, 0.2, 0.3]일 수도 있고 [-100, 200, 300]일 수도 있음 이것을 Internal Covariate Shift라고 하는데, 
    # 학습이 느려지거나, 가중치가 불안정해지고 학습이 잘 안될 수 있음
    # 출력값을 평균 0, 표준편차 1로 정규화하여 학습을 안정화시키는 방법이 Batch Normalization임
    # 무조건 평균 0, 표준편차를 1로 만들어주면, 표현력이 떨어지기 때문에 γ와 β라는 학습 가능한 매개변수를 추가하여, 정규화된 출력값을 다시 스케일링하고 시프트할 수 있도록 함 (y = γ * x̂ + β)
    # 보통 Conv2D와 MaxPooling2D 사이에 BatchNormalization을 추가함, 특히 Conv2D가 점점 커질수록 BatchNormalization의 효과가 더 커짐
    # 단점으로 메모리 사용량 증가, 작은 Batch size에서는 효과 감소
    BatchNormalization(),                   # 각 층의 출력값을 적당한 범위로 정규화하여 학습을 안정화시킴 (과적합 방지에도 도움)
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    
    Dense(512, activation='relu'),
    Dropout(0.5),  # 과적합 방지를 위해 Dropout 추가
    
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 학습을 너무 오래하면 과적합 문제가 발생 할 수 있기 때문에 EarlyStopping 콜백을 사용하여 검증 손실이 N회 연속으로 개선되지 않으면 학습을 중단하고, 가장 좋은 가중치를 복원하도록 설정
# monitor : 검증 손실을 모니터링, patience : 개선이 없는 에포크 수, restore_best_weights : 가장 좋은 가중치를 복원
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(datagen.flow(x_train, y_train, batch_size=256), epochs=100, validation_data=(x_test, y_test), callbacks=[early_stopping])

test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

pred = model.predict(x_test)

result = np.argmax(pred[0])
print(result)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
print("Predicted class :", class_names[result])

# BatchNormalization 사용 전 : 69%, 사용 후 : 72% (기준 10 에포크)
