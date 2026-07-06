# Image Net : 대규모 이미지 데이터 셋, 이미지수 1,000만개 이상, 1,000개 이상의 카테고리
# 입력 이미지 : 224 x 224 x 3 (RGB)
# 보통 직접 학습하지 않고, resnet50, vgg16, inception 등 사전학습된 모델을 가져와서 사용
# ResNet50의 아이디어
#   - Degradation Problem : CNN이 깊어질수록 성능이 떨어지는 문제가 발생
#   - 원인 : Gradient Vanishing, Gradient Exploding (역전파시 앞쪽 레이어까지 제대로 전달되지 않는 문제 발생)
#   - 해결 : Skip Connection을 통해 역전파시 Gradient가 잘 전달되도록 함 (특별한 특징을 학습하지 않으면, 입력을 그대로 출력하도록 학습)


import tensorflow as tf

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import decode_predictions

# ResNet50 모델 불러오기 (사전학습된 가중치 사용)
model = ResNet50(weights='imagenet')

# 이미지 불러오기
img = tf.keras.utils.load_img("lena.png", target_size=(224, 224))

x = tf.keras.utils.img_to_array(img)    # 이미지를 배열로 변환
x = tf.expand_dims(x, axis=0)           # 배치 차원 추가 (모델 입력 형태 맞추기)
x = preprocess_input(x)                 # 전처리 (이미지넷 모델에 맞게)

pred = model.predict(x)

# Top-5 결과 출력
result = decode_predictions(pred, top=5)[0]

print("\n=== Prediction Results ===")

for rank, (imagenetID, label, score) in enumerate(result, start=1):
    print(f"Rank {rank}. {label:25s} " f"{score * 100:.2f}%")