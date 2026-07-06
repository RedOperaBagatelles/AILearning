# scalar (스칼라) : 가장 기본적인 데이터 (0차원, 값 하나만 저장, ex : 5, 3.14, 'A')

import tensorflow as tf

sclar = tf.constant(10)
print(sclar)

# vector (벡터) : 1 차원 벡터 (Rank = 1차원, shape=(3,) == 3개의 원소를 가진 벡터)
vector = tf.constant([1, 2, 3])
print(vector)

# matrix (행렬) : 2 차원 배열 (Rank = 2차원, shape=(3, 2) == 3행 2열의 배열)
matrix = tf.constant([[1, 2], [3, 4], [5, 6]])
print(matrix)

# N-D tensor (3차원 이상의 텐서)
tensor3D = tf.constant([ # 3차원 텐서 (Rank = 3차원, shape=(2, 2, 2) == 2개의 2행 2열의 배열)
    [
        [1, 2],
        [3, 4]
    ],
    [
        [5, 6],
        [7, 8]
    ]
])

print(tensor3D)