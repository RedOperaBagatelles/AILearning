// AILearning.cpp
// MNIST 데이터셋을 불러와 2층 MLP 신경망으로 학습 및 평가하는 프로그램
// 구조: 784(입력) → 256(은닉, ReLU) → 10(출력, Softmax)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <random>

// ============================================================
// 하이퍼파라미터
// ============================================================
constexpr int INPUT_SIZE  = 784;   // 28x28
constexpr int HIDDEN_SIZE = 256;
constexpr int OUTPUT_SIZE = 10;    // 0~9 숫자
constexpr int TRAIN_COUNT = 60000;
constexpr int TEST_COUNT  = 10000;
constexpr int BATCH_SIZE  = 64;
constexpr int EPOCHS      = 10;
constexpr float LEARNING_RATE = 0.01f;

// ============================================================
// 데이터 로딩
// ============================================================
static bool loadImages(const char* path, float* buf, int count)
{
	FILE* fp = nullptr;
	if (fopen_s(&fp, path, "rb") != 0 || !fp)
	{
		printf("Error: Cannot open %s\n", path);
		return false;
	}
	size_t expected = (size_t)count * INPUT_SIZE;
	size_t read = fread(buf, sizeof(float), expected, fp);
	fclose(fp);
	if (read != expected)
	{
		printf("Error: Expected %zu floats, read %zu from %s\n", expected, read, path);
		return false;
	}
	return true;
}

static bool loadLabels(const char* path, uint8_t* buf, int count)
{
	FILE* fp = nullptr;
	if (fopen_s(&fp, path, "rb") != 0 || !fp)
	{
		printf("Error: Cannot open %s\n", path);
		return false;
	}
	size_t read = fread(buf, sizeof(uint8_t), count, fp);
	fclose(fp);
	if (read != (size_t)count)
	{
		printf("Error: Expected %d labels, read %zu from %s\n", count, read, path);
		return false;
	}
	return true;
}

// ============================================================
// 신경망 파라미터
// ============================================================
struct Network
{
	// 가중치 & 편향
	float W1[INPUT_SIZE * HIDDEN_SIZE];   // 784 x 256
	float b1[HIDDEN_SIZE];                // 256
	float W2[HIDDEN_SIZE * OUTPUT_SIZE];  // 256 x 10
	float b2[OUTPUT_SIZE];                // 10

	// 기울기
	float dW1[INPUT_SIZE * HIDDEN_SIZE];
	float db1[HIDDEN_SIZE];
	float dW2[HIDDEN_SIZE * OUTPUT_SIZE];
	float db2[OUTPUT_SIZE];

	// 중간값 (배치 하나분)
	float hidden[BATCH_SIZE * HIDDEN_SIZE];     // ReLU 출력
	float output[BATCH_SIZE * OUTPUT_SIZE];     // Softmax 출력
};

// He 초기화
static void heInit(float* w, int fanIn, int count, std::mt19937& rng)
{
	float stddev = sqrtf(2.0f / (float)fanIn);
	std::normal_distribution<float> dist(0.0f, stddev);
	for (int i = 0; i < count; i++)
		w[i] = dist(rng);
}

static void initNetwork(Network& net)
{
	std::mt19937 rng(42);  // 재현성을 위한 고정 시드
	heInit(net.W1, INPUT_SIZE,  INPUT_SIZE * HIDDEN_SIZE, rng);
	heInit(net.W2, HIDDEN_SIZE, HIDDEN_SIZE * OUTPUT_SIZE, rng);
	memset(net.b1, 0, sizeof(net.b1));
	memset(net.b2, 0, sizeof(net.b2));
}

// ============================================================
// Forward Pass
// ============================================================
static void forward(Network& net, const float* X, int batchSize)
{
	// Layer 1: Z1 = X * W1 + b1, H = ReLU(Z1)
	for (int b = 0; b < batchSize; b++)
	{
		const float* x = X + b * INPUT_SIZE;
		float* h = net.hidden + b * HIDDEN_SIZE;

		for (int j = 0; j < HIDDEN_SIZE; j++)
		{
			float sum = net.b1[j];
			for (int i = 0; i < INPUT_SIZE; i++)
				sum += x[i] * net.W1[i * HIDDEN_SIZE + j];
			h[j] = sum > 0.0f ? sum : 0.0f; // ReLU
		}
	}

	// Layer 2: Z2 = H * W2 + b2, Y = Softmax(Z2)
	for (int b = 0; b < batchSize; b++)
	{
		const float* h = net.hidden + b * HIDDEN_SIZE;
		float* o = net.output + b * OUTPUT_SIZE;

		// 선형 변환
		for (int j = 0; j < OUTPUT_SIZE; j++)
		{
			float sum = net.b2[j];
			for (int i = 0; i < HIDDEN_SIZE; i++)
				sum += h[i] * net.W2[i * OUTPUT_SIZE + j];
			o[j] = sum;
		}

		// Softmax (수치 안정성을 위해 max 빼기)
		float maxVal = o[0];
		for (int j = 1; j < OUTPUT_SIZE; j++)
			if (o[j] > maxVal) maxVal = o[j];

		float sumExp = 0.0f;
		for (int j = 0; j < OUTPUT_SIZE; j++)
		{
			o[j] = expf(o[j] - maxVal);
			sumExp += o[j];
		}
		for (int j = 0; j < OUTPUT_SIZE; j++)
			o[j] /= sumExp;
	}
}

// ============================================================
// Backward Pass (역전파) + Cross-Entropy Loss
// ============================================================
static float backward(Network& net, const float* X, const uint8_t* labels, int batchSize)
{
	float totalLoss = 0.0f;

	// 기울기 초기화
	memset(net.dW1, 0, sizeof(net.dW1));
	memset(net.db1, 0, sizeof(net.db1));
	memset(net.dW2, 0, sizeof(net.dW2));
	memset(net.db2, 0, sizeof(net.db2));

	float invBatch = 1.0f / (float)batchSize;

	for (int b = 0; b < batchSize; b++)
	{
		const float* x = X + b * INPUT_SIZE;
		const float* h = net.hidden + b * HIDDEN_SIZE;
		float* o = net.output + b * OUTPUT_SIZE;
		int label = labels[b];

		// Cross-entropy 손실
		float prob = o[label] > 1e-7f ? o[label] : 1e-7f;
		totalLoss -= logf(prob);

		// 출력층 기울기: dL/dZ2 = softmax_output - one_hot(label)
		float dZ2[OUTPUT_SIZE];
		for (int j = 0; j < OUTPUT_SIZE; j++)
			dZ2[j] = (o[j] - (j == label ? 1.0f : 0.0f)) * invBatch;

		// W2 기울기: dW2 += h^T * dZ2
		for (int i = 0; i < HIDDEN_SIZE; i++)
			for (int j = 0; j < OUTPUT_SIZE; j++)
				net.dW2[i * OUTPUT_SIZE + j] += h[i] * dZ2[j];

		// b2 기울기
		for (int j = 0; j < OUTPUT_SIZE; j++)
			net.db2[j] += dZ2[j];

		// 은닉층 기울기: dH = dZ2 * W2^T, dZ1 = dH * (h > 0)
		float dH[HIDDEN_SIZE];
		for (int i = 0; i < HIDDEN_SIZE; i++)
		{
			float sum = 0.0f;
			for (int j = 0; j < OUTPUT_SIZE; j++)
				sum += dZ2[j] * net.W2[i * OUTPUT_SIZE + j];
			dH[i] = (h[i] > 0.0f) ? sum : 0.0f; // ReLU 미분
		}

		// W1 기울기: dW1 += x^T * dH
		for (int i = 0; i < INPUT_SIZE; i++)
			for (int j = 0; j < HIDDEN_SIZE; j++)
				net.dW1[i * HIDDEN_SIZE + j] += x[i] * dH[j];

		// b1 기울기
		for (int j = 0; j < HIDDEN_SIZE; j++)
			net.db1[j] += dH[j];
	}

	return totalLoss / (float)batchSize;
}

// ============================================================
// SGD 파라미터 업데이트
// ============================================================
static void sgdUpdate(Network& net, float lr)
{
	for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
		net.W1[i] -= lr * net.dW1[i];
	for (int i = 0; i < HIDDEN_SIZE; i++)
		net.b1[i] -= lr * net.db1[i];
	for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
		net.W2[i] -= lr * net.dW2[i];
	for (int i = 0; i < OUTPUT_SIZE; i++)
		net.b2[i] -= lr * net.db2[i];
}

// ============================================================
// 평가 (정확도 계산)
// ============================================================
static float evaluate(Network& net, const float* images, const uint8_t* labels, int count)
{
	int correct = 0;

	// 배치 단위로 처리
	for (int offset = 0; offset < count; offset += BATCH_SIZE)
	{
		int bs = (offset + BATCH_SIZE <= count) ? BATCH_SIZE : (count - offset);
		forward(net, images + offset * INPUT_SIZE, bs);

		for (int b = 0; b < bs; b++)
		{
			const float* o = net.output + b * OUTPUT_SIZE;
			int pred = 0;
			float maxVal = o[0];
			for (int j = 1; j < OUTPUT_SIZE; j++)
			{
				if (o[j] > maxVal)
				{
					maxVal = o[j];
					pred = j;
				}
			}
			if (pred == labels[offset + b])
				correct++;
		}
	}

	return (float)correct / (float)count * 100.0f;
}

// ============================================================
// 메인
// ============================================================
int main()
{
	printf("=== MNIST Neural Network Training (Pure C++) ===\n");
	printf("Architecture: %d -> %d (ReLU) -> %d (Softmax)\n", INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
	printf("Batch Size: %d, Epochs: %d, Learning Rate: %.4f\n\n", BATCH_SIZE, EPOCHS, LEARNING_RATE);

	// ----- 데이터 로딩 -----
	printf("[1/3] Loading MNIST data...\n");

	static float trainImages[TRAIN_COUNT * INPUT_SIZE];
	static uint8_t trainLabels[TRAIN_COUNT];
	static float testImages[TEST_COUNT * INPUT_SIZE];
	static uint8_t testLabels[TEST_COUNT];

	if (!loadImages("mnist_train_images.bin", trainImages, TRAIN_COUNT)) return 1;
	if (!loadLabels("mnist_train_labels.bin", trainLabels, TRAIN_COUNT)) return 1;
	if (!loadImages("mnist_test_images.bin",  testImages,  TEST_COUNT))  return 1;
	if (!loadLabels("mnist_test_labels.bin",  testLabels,  TEST_COUNT))  return 1;

	printf("  Train: %d images, Test: %d images\n\n", TRAIN_COUNT, TEST_COUNT);

	// ----- 네트워크 초기화 -----
	printf("[2/3] Initializing network...\n\n");
	static Network net;
	initNetwork(net);

	// ----- 학습 -----
	printf("[3/3] Training...\n");
	printf("-----------------------------------------------\n");
	printf("  Epoch  |   Loss   | Train Acc | Test Acc\n");
	printf("-----------------------------------------------\n");

	std::mt19937 rng(123);
	std::vector<int> indices(TRAIN_COUNT);
	for (int i = 0; i < TRAIN_COUNT; i++) indices[i] = i;

	// 배치용 임시 버퍼
	static float batchImages[BATCH_SIZE * INPUT_SIZE];
	static uint8_t batchLabels[BATCH_SIZE];

	clock_t totalStart = clock();

	for (int epoch = 0; epoch < EPOCHS; epoch++)
	{
		clock_t epochStart = clock();

		// 셔플
		std::shuffle(indices.begin(), indices.end(), rng);

		float epochLoss = 0.0f;
		int batchCount = 0;

		for (int offset = 0; offset + BATCH_SIZE <= TRAIN_COUNT; offset += BATCH_SIZE)
		{
			// 미니배치 구성
			for (int b = 0; b < BATCH_SIZE; b++)
			{
				int idx = indices[offset + b];
				memcpy(batchImages + b * INPUT_SIZE,
				       trainImages + idx * INPUT_SIZE,
				       INPUT_SIZE * sizeof(float));
				batchLabels[b] = trainLabels[idx];
			}

			// Forward → Backward → Update
			forward(net, batchImages, BATCH_SIZE);
			float loss = backward(net, batchImages, batchLabels, BATCH_SIZE);
			sgdUpdate(net, LEARNING_RATE);

			epochLoss += loss;
			batchCount++;
		}

		epochLoss /= (float)batchCount;
		float trainAcc = evaluate(net, trainImages, trainLabels, TRAIN_COUNT);
		float testAcc  = evaluate(net, testImages,  testLabels,  TEST_COUNT);

		double elapsed = (double)(clock() - epochStart) / CLOCKS_PER_SEC;

		printf("  %2d/%2d  |  %.4f  |  %5.2f%%  |  %5.2f%%   (%.1fs)\n",
			epoch + 1, EPOCHS, epochLoss, trainAcc, testAcc, elapsed);
	}

	printf("-----------------------------------------------\n");

	double totalTime = (double)(clock() - totalStart) / CLOCKS_PER_SEC;
	float finalTestAcc = evaluate(net, testImages, testLabels, TEST_COUNT);
	printf("\nFinal Test Accuracy: %.2f%%\n", finalTestAcc);
	printf("Total Training Time: %.1f seconds\n", totalTime);
	printf("\nDone!\n");

	return 0;
}