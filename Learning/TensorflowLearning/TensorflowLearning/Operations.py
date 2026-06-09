import tensorflow as tf

a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

# tensorflow는 기본적인 사칙 연산을 지원함
print(a + b)
print(a - b)
print(a * b)
print(a / b)

# 메소드로 사용 가능
c = tf.add(a, b)
print(c)

c = tf.subtract(a, b)
print(c)

c = tf.multiply(a, b)
print(c)

c = tf.divide(a, b)
print(c)

a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])

# 행렬 곱도 가능
c = tf.matmul(a, b)
print(c)

# broadcasting : 크기가 다른 센서를 자동으로 확장하여 연산하는 기능
a = tf.constant([[1, 2, 3], [4, 5, 6]])
b = tf.constant([10, 20, 30])

# b를 [[10, 20, 30], [10, 20, 30]]으로 내부적으로 확장시켜 계산함
print(a + b)

# broadcasting 규칙
#   - (2, 3) (3, )처럼 두 차원이 있을 때 앞에 1을 붙어줌 (2, 3) (1, 3), 이후 맨 뒤 차원인 3 == 3이므로 1 → 2로 확장 가능 (2, 3)
#   - (2, 3) (2, 1)인 경우 (2, 3) (2, 3)으로 확장 가능
#   - (2, 3) (4, )인 경우 먼저 1을 앞에 붙이면 (2, 3) (1, 4)가 되는데 이 경우 맨 뒤 차원이 3 != 4이므로 확장 불가능

# reshape : 텐서의 모양을 변경해줌
a = tf.constant([1, 2, 3, 4, 5, 6])
b = tf.reshape(a, [2, 3])           # shape=(2, 3) 형태로 모양이 변경됨
print(b)

b = tf. reshape(a, [3, 2])          # shape=(3, 2) 형태로 모양이 변경됨
print(b)

# reshape -1 사용 : tensorflow가 자동으로 원소의 개수를 맞춰줌
b = tf.reshape(a, [-1, 3])  # 자동으로 -1 부분을 2로 설정해줌
print(b)

# reshape 주의점 : 원소의 개수는 동일해야 함
b = tf.reshape(a, [6])  # succeed
# b = tf.reshape(a, [7])  error
print(b)

# 행과 열을 뒤집음
b = tf.reshape(a, [-1, 3])
b = tf.transpose(b)
print(b)

# 다차원 Transpose (perm에 따라서 차원의 순서를 지정할 수 있음)
a = tf.random.normal([1, 2, 3, 4])
a = tf.transpose(a, perm=[2, 0, 1, 3])  # shape=(3, 1, 2, 4)로 순서가 바뀜
print(a)