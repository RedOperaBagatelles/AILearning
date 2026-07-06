from tensorflow.keras.datasets import mnist

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

# tensorflow로부터 MNIST의 데이터 셋을 가져옴
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) # (60000, 28, 28) 60000장의 이미지, 이미지 크기 28 x 28
print(y_train.shape) # (60000,)

# 모든 픽셀 값 범위를 0 ~ 1로 정규화 함
x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN을 사용할 경우 채널을 추가해야 함 (60000, 28, 28) → (60000, 28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# 모델 추가
model = Sequential([
    Input(shape=(28, 28, 1)),                   # 입력층 (28 x 28 크기의 흑백 이미지[채널이 1개, 컬러이면 3개])
    
    # 특정을 추출하기 위해서 32개의 필터를 사용, 보통 CNN에서는 2의 제곱수를 사용, 너무 크면 연산량이 많아지고, 너무 작으면 특징을 잘 추출하지 못함 (기본적인 특징만 추출)
    Conv2D(32, (3, 3), activation='relu'),      # 3 x 3 크기의 필터를 사용 (1x1는 너무 작고, 5x5는 너무 커서 3x3을 많이 사용함, 보통 3x3 필터나 5x5 필터를 사용함)
    MaxPool2D((2, 2)),                          # 2 x 2 크기의 풀링을 사용 (2 x 2 크기의 영역에서 가장 큰 값을 추출, 특징을 추출하고, 연산량을 줄이기 위해서 사용, 2 x 2가 가장 많이 사용됨)
    
    # 특정을 추출하기 위해서 64개의 필터를 사용, 더 복잡한 특징을 추출하기 위해서 필터의 개수를 늘림
    Conv2D(64, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),
    
    # ==================
    # 여기에 더 높은 수준의 특징을 추출하기 위해서 Conv2D와 MaxPool2D를 추가할 수 있음 (너무 많이 추가하면 모델이 너무 복잡해져 훈련 데이터를 외워버리는 과적합이 발생할 수 있음)
    # ===================
    
    Flatten(),                                  # 2 차원 데이터를 1차원으로 변환 (CNN에서 추출한 특징을 Dense층에 전달하기 위해서 사용)
    
    Dense(128, activation='relu'),              # 은닉층 (128개의 뉴런을 사용, 128은 보통 2의 제곱수를 사용, 너무 크면 연산량이 많아지고, 너무 작으면 특징을 잘 추출하지 못함)
    Dense(10, activation='softmax')             # 출력층 (10개의 뉴런을 사용, 0~9까지의 숫자를 분류하기 위해서 10개의 뉴런을 사용, softmax는 다중 분류용 활성화 함수로, 각 클래스에 대한 확률을 출력함)
])

# 모델 컴파일 (가중치 최적화, 손실함수, 평가 지표 설정, sparse_categorical_crossentropy는 다중 분류용)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습 (x_train, y_train 데이터로 학습, 반복 회수 5회, 32개씩 나누어 학습, 검증 데이터 20% 사용)
history = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# 모델 평가 (x_test, y_test 데이터로 평가)
loss, acc = model.evaluate(x_test, y_test)

print(acc)

# 모델을 이용하여 테스트 데이터에 대한 예측 수행
pred = model.predict(x_test[:1])

# 각 클래스에 대한 확률을 출력(% 단위로 표시)
for i in range(10):
    print(f"클래스 {i}: {pred[0][i]*100:.2f}%")

# 예측 결과를 출력 (가장 높은 확률을 가진 클래스)
result = np.argmax(pred)
print(result)

plt.imshow(x_test[0].reshape(28, 28), cmap='gray')
plt.show()