// MnistTensorflow.cpp
// TensorFlow를 사용한 MNIST 데이터셋 학습 및 평가
// 구조: 784(입력) → 256(은닉, ReLU) → 10(출력, Softmax)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>

#include "tensorflow/include/c/c_api.h"

// ============================================================
// 하이퍼파라미터
// ============================================================
constexpr int inputSize = 784;           // 28x28
constexpr int hiddenSize = 256;          // 은닉층 뉴런 수
constexpr int outputSize = 10;           // 출력층 뉴런 수 (0 ~ 9)
constexpr int trainCount = 60000;        // 훈련 데이터 수
constexpr int testCount = 10000;         // 테스트 데이터 수
constexpr int batchCount = 64;           // 배치 크기
constexpr int EPOCHS = 10000;            // 학습 반복 횟수
constexpr float LEARNING_RATE = 0.001f;  // 학습률

// ============================================================
// 에러 처리 헬퍼
// ============================================================
static void CheckStatus(TF_Status* status)
{
    if (TF_GetCode(status) != TF_OK)
    {
        fprintf(stderr, "Error: %s\n", TF_Message(status));
        TF_DeleteStatus(status);
        exit(1);
    }
}

// ============================================================
// 데이터 로딩
// ============================================================
static bool LoadImages(const char* path, float* buf, int count)
{
    FILE* fp = nullptr;

    if (fopen_s(&fp, path, "rb") != 0 || fp == nullptr)
    {
        printf("Error: Cannot open %s\n", path);
        return false;
    }

    size_t expected = (size_t)count * inputSize;
    size_t read = fread(buf, sizeof(float), expected, fp);

    fclose(fp);

    if (read != expected)
    {
        printf("Error: Expected %zu floats, read %zu from %s\n", expected, read, path);
        return false;
    }

    return true;
}

static bool LoadLabels(const char* path, uint8_t* buf, int count)
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
// He 초기화
// ============================================================
static void HeInit(float* w, int fanIn, int count, std::mt19937& rng)
{
    float stddev = sqrtf(2.0f / (float)fanIn);
    std::normal_distribution<float> dist(0.0f, stddev);

    for (int i = 0; i < count; i++)
        w[i] = dist(rng);
}

struct GraphOps
{
    TF_Output input;
    TF_Output labels;
    TF_Output weight1;
    TF_Output bias1;
    TF_Output weight2;
    TF_Output bias2;

    TF_Output loss;
    TF_Output accuracy;

    TF_Output gradW1;
    TF_Output gradB1;
    TF_Output gradW2;
    TF_Output gradB2;
};

// ============================================================
// 신경망 구축
// ============================================================
static TF_Graph* BuildGraph(GraphOps& ops)
{
    TF_Graph* graph = TF_NewGraph();
    TF_Status* status = TF_NewStatus();

    auto CreatePlaceholder = [&](const char* name, TF_DataType dtype, const int64_t* shape, int dims) -> TF_Output
    {
        TF_OperationDescription* d = TF_NewOperation(graph, "Placeholder", name);
        TF_SetAttrType(d, "dtype", dtype);
        TF_SetAttrShape(d, "shape", shape, dims);

        TF_Output out;
        out.oper = TF_FinishOperation(d, status);
        out.index = 0;
        CheckStatus(status);

        return out;
    };

    auto CreateScalarI32Const = [&](const char* name, int32_t value) -> TF_Output
    {
        TF_Tensor* tensor = TF_AllocateTensor(TF_INT32, nullptr, 0, sizeof(value));
        memcpy(TF_TensorData(tensor), &value, sizeof(value));

        TF_OperationDescription* d = TF_NewOperation(graph, "Const", name);
        TF_SetAttrTensor(d, "value", tensor, status);
        CheckStatus(status);
        TF_SetAttrType(d, "dtype", TF_INT32);

        TF_Output out;
        out.oper = TF_FinishOperation(d, status);
        out.index = 0;
        CheckStatus(status);

        TF_DeleteTensor(tensor);
        return out;
    };

    int64_t inputShape[] = { -1, inputSize };
    int64_t labelShape[] = { -1, outputSize };
    int64_t w1Shape[] = { inputSize, hiddenSize };
    int64_t b1Shape[] = { hiddenSize };
    int64_t w2Shape[] = { hiddenSize, outputSize };
    int64_t b2Shape[] = { outputSize };

    // 입력/라벨/가중치 placeholder
    ops.input = CreatePlaceholder("input", TF_FLOAT, inputShape, 2);
    ops.labels = CreatePlaceholder("labels", TF_FLOAT, labelShape, 2);
    ops.weight1 = CreatePlaceholder("weight1", TF_FLOAT, w1Shape, 2);
    ops.bias1 = CreatePlaceholder("bias1", TF_FLOAT, b1Shape, 1);
    ops.weight2 = CreatePlaceholder("weight2", TF_FLOAT, w2Shape, 2);
    ops.bias2 = CreatePlaceholder("bias2", TF_FLOAT, b2Shape, 1);

    TF_OperationDescription* desc = nullptr;

    // Hidden layer: MatMul(input, weight1) + bias1, then ReLU
    desc = TF_NewOperation(graph, "MatMul", "matmul1");
    TF_AddInput(desc, ops.input);
    TF_AddInput(desc, ops.weight1);
    TF_Output matmul1Out;
    matmul1Out.oper = TF_FinishOperation(desc, status);
    matmul1Out.index = 0;
    CheckStatus(status);

    desc = TF_NewOperation(graph, "BiasAdd", "biasadd1");
    TF_AddInput(desc, matmul1Out);
    TF_AddInput(desc, ops.bias1);
    TF_Output biasadd1Out;
    biasadd1Out.oper = TF_FinishOperation(desc, status);
    biasadd1Out.index = 0;
    CheckStatus(status);

    desc = TF_NewOperation(graph, "Relu", "relu1");
    TF_AddInput(desc, biasadd1Out);
    TF_Output reluOut;
    reluOut.oper = TF_FinishOperation(desc, status);
    reluOut.index = 0;
    CheckStatus(status);

    // Output layer: MatMul(hidden, weight2) + bias2
    desc = TF_NewOperation(graph, "MatMul", "matmul2");
    TF_AddInput(desc, reluOut);
    TF_AddInput(desc, ops.weight2);
    TF_Output matmul2Out;
    matmul2Out.oper = TF_FinishOperation(desc, status);
    matmul2Out.index = 0;
    CheckStatus(status);

    desc = TF_NewOperation(graph, "BiasAdd", "biasadd2");
    TF_AddInput(desc, matmul2Out);
    TF_AddInput(desc, ops.bias2);
    TF_Output logitsOut;
    logitsOut.oper = TF_FinishOperation(desc, status);
    logitsOut.index = 0;
    CheckStatus(status);

    // Softmax
    desc = TF_NewOperation(graph, "Softmax", "softmax");
    TF_AddInput(desc, logitsOut);
    TF_Output softmaxOut;
    softmaxOut.oper = TF_FinishOperation(desc, status);
    softmaxOut.index = 0;
    CheckStatus(status);

    // Loss: CrossEntropyWithSoftmax (one-hot labels)
    desc = TF_NewOperation(graph, "SoftmaxCrossEntropyWithLogits", "loss");
    TF_SetAttrType(desc, "T", TF_FLOAT);
    TF_AddInput(desc, logitsOut);
    TF_AddInput(desc, ops.labels);
    TF_Output lossPerBatchOut;
    lossPerBatchOut.oper = TF_FinishOperation(desc, status);
    lossPerBatchOut.index = 0;
    CheckStatus(status);

    TF_Output axis0Out = CreateScalarI32Const("axis0", 0);
    TF_Output axis1Out = CreateScalarI32Const("axis1", 1);

    // Mean loss
    desc = TF_NewOperation(graph, "Mean", "mean_loss");
    TF_AddInput(desc, lossPerBatchOut);
    TF_AddInput(desc, axis0Out);
    ops.loss.oper = TF_FinishOperation(desc, status);
    ops.loss.index = 0;
    CheckStatus(status);

    // Accuracy
    desc = TF_NewOperation(graph, "ArgMax", "predictions");
    TF_AddInput(desc, softmaxOut);
    TF_AddInput(desc, axis1Out);
    TF_SetAttrType(desc, "output_type", TF_INT64);
    TF_Output predictionsOut;
    predictionsOut.oper = TF_FinishOperation(desc, status);
    predictionsOut.index = 0;
    CheckStatus(status);

    desc = TF_NewOperation(graph, "ArgMax", "label_ids");
    TF_AddInput(desc, ops.labels);
    TF_AddInput(desc, axis1Out);
    TF_SetAttrType(desc, "output_type", TF_INT64);
    TF_Output labelIdsOut;
    labelIdsOut.oper = TF_FinishOperation(desc, status);
    labelIdsOut.index = 0;
    CheckStatus(status);

    desc = TF_NewOperation(graph, "Equal", "correct");
    TF_AddInput(desc, predictionsOut);
    TF_AddInput(desc, labelIdsOut);
    TF_Output correctOut;
    correctOut.oper = TF_FinishOperation(desc, status);
    correctOut.index = 0;
    CheckStatus(status);

    desc = TF_NewOperation(graph, "Cast", "correct_float");
    TF_AddInput(desc, correctOut);
    TF_SetAttrType(desc, "DstT", TF_FLOAT);
    TF_Output correctFloatOut;
    correctFloatOut.oper = TF_FinishOperation(desc, status);
    correctFloatOut.index = 0;
    CheckStatus(status);

    desc = TF_NewOperation(graph, "Mean", "accuracy");
    TF_AddInput(desc, correctFloatOut);
    TF_AddInput(desc, axis0Out);
    ops.accuracy.oper = TF_FinishOperation(desc, status);
    ops.accuracy.index = 0;
    CheckStatus(status);

    TF_Output ys[1] = { ops.loss };
    TF_Output xs[4] = { ops.weight1, ops.bias1, ops.weight2, ops.bias2 };
    TF_Output grads[4];
    TF_AddGradients(graph, ys, 1, xs, 4, nullptr, status, grads);
    CheckStatus(status);

    ops.gradW1 = grads[0];
    ops.gradB1 = grads[1];
    ops.gradW2 = grads[2];
    ops.gradB2 = grads[3];

    TF_DeleteStatus(status);

    return graph;
}

// ============================================================
// 배치 데이터 준비
// ============================================================
static TF_Tensor* CreateTensor(TF_DataType type, const int64_t* dims, int nDims, void* data, size_t dataSize)
{
    TF_Tensor* tensor = TF_AllocateTensor(type, dims, nDims, dataSize);
    if (data)
        memcpy(TF_TensorData(tensor), data, dataSize);
    return tensor;
}

// ============================================================
// Forward Pass (Inference)
// ============================================================
static float Evaluate(TF_Session* session, const GraphOps& ops,
    const float* images, const uint8_t* labels, int count,
    const float* weight1, const float* b1, const float* weight2, const float* b2,
    float& outLoss)
{
    TF_Status* status = TF_NewStatus();
    int correct = 0;
    float totalLoss = 0.0f;

    for (int offset = 0; offset < count; offset += batchCount)
    {
        int bs = (offset + batchCount <= count) ? batchCount : (count - offset);

        int64_t inputShape[] = { bs, inputSize };
        int64_t labelShape[] = { bs, outputSize };
        int64_t w1Shape[] = { inputSize, hiddenSize };
        int64_t b1Shape[] = { hiddenSize };
        int64_t w2Shape[] = { hiddenSize, outputSize };
        int64_t b2Shape[] = { outputSize };

        TF_Tensor* inputTensor = CreateTensor(TF_FLOAT, inputShape, 2,
            (void*)(images + offset * inputSize), (size_t)bs * inputSize * sizeof(float));

        std::vector<float> labelData((size_t)bs * outputSize, 0.0f);
        for (int i = 0; i < bs; i++)
            labelData[(size_t)i * outputSize + labels[offset + i]] = 1.0f;
        TF_Tensor* labelTensor = CreateTensor(TF_FLOAT, labelShape, 2,
            labelData.data(), (size_t)bs * outputSize * sizeof(float));

        TF_Tensor* w1Tensor = CreateTensor(TF_FLOAT, w1Shape, 2,
            (void*)weight1, sizeof(float) * inputSize * hiddenSize);
        TF_Tensor* b1Tensor = CreateTensor(TF_FLOAT, b1Shape, 1,
            (void*)b1, sizeof(float) * hiddenSize);
        TF_Tensor* w2Tensor = CreateTensor(TF_FLOAT, w2Shape, 2,
            (void*)weight2, sizeof(float) * hiddenSize * outputSize);
        TF_Tensor* b2Tensor = CreateTensor(TF_FLOAT, b2Shape, 1,
            (void*)b2, sizeof(float) * outputSize);

        TF_Output inputs[] = { ops.input, ops.labels, ops.weight1, ops.bias1, ops.weight2, ops.bias2 };
        TF_Tensor* inputValues[] = { inputTensor, labelTensor, w1Tensor, b1Tensor, w2Tensor, b2Tensor };

        TF_Output outputs[] = { ops.accuracy, ops.loss };
        TF_Tensor* outputValues[] = { nullptr, nullptr };

        TF_SessionRun(session, nullptr,
            inputs, inputValues, 6,
            outputs, outputValues, 2,
            nullptr, 0, nullptr, status);
        CheckStatus(status);

        const float acc = ((float*)TF_TensorData(outputValues[0]))[0];
        const float loss = ((float*)TF_TensorData(outputValues[1]))[0];

        correct += (int)(acc * bs + 0.5f);
        totalLoss += loss * bs;

        TF_DeleteTensor(inputTensor);
        TF_DeleteTensor(labelTensor);
        TF_DeleteTensor(w1Tensor);
        TF_DeleteTensor(b1Tensor);
        TF_DeleteTensor(w2Tensor);
        TF_DeleteTensor(b2Tensor);
        TF_DeleteTensor(outputValues[0]);
        TF_DeleteTensor(outputValues[1]);
    }

    TF_DeleteStatus(status);

    outLoss = totalLoss / (float)count;
    return (float)correct / (float)count * 100.0f;
}

// ============================================================
// 메인
// ============================================================
//int main()
//{
//    printf("=== MNIST Neural Network Training (TensorFlow C API) ===\n");
//    printf("Architecture: %d -> %d (ReLU) -> %d (Softmax)\n", inputSize, hiddenSize, outputSize);
//    printf("Batch Size: %d, Epochs: %d, Learning Rate: %.4f\n\n", batchCount, EPOCHS, LEARNING_RATE);
//
//    // ----- 데이터 로딩 -----
//    printf("[1/4] Loading MNIST data...\n");
//
//    static float trainImages[trainCount * inputSize];
//    static uint8_t trainLabels[trainCount];
//    static float testImages[testCount * inputSize];
//    static uint8_t testLabels[testCount];
//
//    if (!LoadImages("mnist_train_images.bin", trainImages, trainCount)) return 1;
//    if (!LoadLabels("mnist_train_labels.bin", trainLabels, trainCount)) return 1;
//    if (!LoadImages("mnist_test_images.bin", testImages, testCount)) return 1;
//    if (!LoadLabels("mnist_test_labels.bin", testLabels, testCount)) return 1;
//
//    printf("  Train: %d images, Test: %d images\n\n", trainCount, testCount);
//
//    // ----- 네트워크 초기화 -----
//    printf("[2/4] Initializing network...\n");
//
//    std::mt19937 random(42);
//    float weight1[inputSize * hiddenSize];
//    float b1[hiddenSize];
//    float weight2[hiddenSize * outputSize];
//    float b2[outputSize];
//
//    HeInit(weight1, inputSize, inputSize * hiddenSize, random);
//    HeInit(weight2, hiddenSize, hiddenSize * outputSize, random);
//
//    memset(b1, 0, sizeof(b1));
//    memset(b2, 0, sizeof(b2));
//
//    printf("  Network initialized\n\n");
//
//    // ----- 그래프 구축 -----
//    printf("[3/4] Building TensorFlow graph...\n");
//
//    GraphOps ops;
//    TF_Graph* graph = BuildGraph(ops);
//
//    TF_SessionOptions* options = TF_NewSessionOptions();
//    TF_Status* status = TF_NewStatus();
//    TF_Session* session = TF_NewSession(graph, options, status);
//    CheckStatus(status);
//
//    printf("  Graph and session created\n\n");
//
//    // ----- 학습 -----
//    printf("[4/4] Training...\n");
//    printf("-----------------------------------------------\n");
//    printf("  Epoch  |   Loss   | Train Acc | Test Acc\n");
//    printf("-----------------------------------------------\n");
//
//    std::mt19937 rng(123);
//    std::vector<int> indices(trainCount);
//    for (int i = 0; i < trainCount; i++)
//        indices[i] = i;
//
//    static float batchImages[batchCount * inputSize];
//    static uint8_t batchLabels[batchCount];
//
//    TF_Status* runStatus = TF_NewStatus();
//    clock_t totalStart = clock();
//
//    for (int epoch = 0; epoch < EPOCHS; epoch++)
//    {
//        clock_t epochStart = clock();
//        std::shuffle(indices.begin(), indices.end(), rng);
//
//        float epochLoss = 0.0f;
//        int batchNum = 0;
//
//        for (int offset = 0; offset + batchCount <= trainCount; offset += batchCount)
//        {
//            for (int b = 0; b < batchCount; b++)
//            {
//                int index = indices[offset + b];
//                memcpy(batchImages + b * inputSize, trainImages + index * inputSize, sizeof(float) * inputSize);
//                batchLabels[b] = trainLabels[index];
//            }
//
//            int64_t inputShape[] = { batchCount, inputSize };
//            int64_t labelShape[] = { batchCount, outputSize };
//            int64_t w1Shape[] = { inputSize, hiddenSize };
//            int64_t b1Shape[] = { hiddenSize };
//            int64_t w2Shape[] = { hiddenSize, outputSize };
//            int64_t b2Shape[] = { outputSize };
//
//            std::vector<float> labelData((size_t)batchCount * outputSize, 0.0f);
//            for (int i = 0; i < batchCount; i++)
//                labelData[(size_t)i * outputSize + batchLabels[i]] = 1.0f;
//
//            TF_Tensor* inputTensor = CreateTensor(TF_FLOAT, inputShape, 2,
//                batchImages, sizeof(float) * batchCount * inputSize);
//            TF_Tensor* labelTensor = CreateTensor(TF_FLOAT, labelShape, 2,
//                labelData.data(), sizeof(float) * batchCount * outputSize);
//            TF_Tensor* w1Tensor = CreateTensor(TF_FLOAT, w1Shape, 2,
//                weight1, sizeof(float) * inputSize * hiddenSize);
//            TF_Tensor* b1Tensor = CreateTensor(TF_FLOAT, b1Shape, 1,
//                b1, sizeof(float) * hiddenSize);
//            TF_Tensor* w2Tensor = CreateTensor(TF_FLOAT, w2Shape, 2,
//                weight2, sizeof(float) * hiddenSize * outputSize);
//            TF_Tensor* b2Tensor = CreateTensor(TF_FLOAT, b2Shape, 1,
//                b2, sizeof(float) * outputSize);
//
//            TF_Output inputs[] = { ops.input, ops.labels, ops.weight1, ops.bias1, ops.weight2, ops.bias2 };
//            TF_Tensor* inputValues[] = { inputTensor, labelTensor, w1Tensor, b1Tensor, w2Tensor, b2Tensor };
//
//            TF_Output outputs[] = { ops.loss, ops.gradW1, ops.gradB1, ops.gradW2, ops.gradB2 };
//            TF_Tensor* outputValues[] = { nullptr, nullptr, nullptr, nullptr, nullptr };
//
//            TF_SessionRun(session, nullptr,
//                inputs, inputValues, 6,
//                outputs, outputValues, 5,
//                nullptr, 0, nullptr, runStatus);
//            CheckStatus(runStatus);
//
//            const float loss = ((float*)TF_TensorData(outputValues[0]))[0];
//            const float* gradW1 = (const float*)TF_TensorData(outputValues[1]);
//            const float* gradB1 = (const float*)TF_TensorData(outputValues[2]);
//            const float* gradW2 = (const float*)TF_TensorData(outputValues[3]);
//            const float* gradB2 = (const float*)TF_TensorData(outputValues[4]);
//
//            for (int i = 0; i < inputSize * hiddenSize; i++)
//                weight1[i] -= LEARNING_RATE * gradW1[i];
//            for (int i = 0; i < hiddenSize; i++)
//                b1[i] -= LEARNING_RATE * gradB1[i];
//            for (int i = 0; i < hiddenSize * outputSize; i++)
//                weight2[i] -= LEARNING_RATE * gradW2[i];
//            for (int i = 0; i < outputSize; i++)
//                b2[i] -= LEARNING_RATE * gradB2[i];
//
//            epochLoss += loss;
//            batchNum++;
//
//            TF_DeleteTensor(inputTensor);
//            TF_DeleteTensor(labelTensor);
//            TF_DeleteTensor(w1Tensor);
//            TF_DeleteTensor(b1Tensor);
//            TF_DeleteTensor(w2Tensor);
//            TF_DeleteTensor(b2Tensor);
//            for (TF_Tensor* t : outputValues)
//                TF_DeleteTensor(t);
//        }
//
//        epochLoss /= (float)batchNum;
//
//        float trainLoss = 0.0f;
//        float testLoss = 0.0f;
//        float trainAcc = Evaluate(session, ops, trainImages, trainLabels, trainCount, weight1, b1, weight2, b2, trainLoss);
//        float testAcc = Evaluate(session, ops, testImages, testLabels, testCount, weight1, b1, weight2, b2, testLoss);
//
//        double elapsed = (double)(clock() - epochStart) / CLOCKS_PER_SEC;
//        printf("  %2d/%2d  |  %.4f  |  %5.2f%%  |  %5.2f%%   (%.1fs)\n",
//            epoch + 1, EPOCHS, epochLoss, trainAcc, testAcc, elapsed);
//    }
//
//    double totalTime = (double)(clock() - totalStart) / CLOCKS_PER_SEC;
//    float finalTestLoss = 0.0f;
//    float finalTestAcc = Evaluate(session, ops, testImages, testLabels, testCount, weight1, b1, weight2, b2, finalTestLoss);
//
//    printf("-----------------------------------------------\n");
//    printf("\nFinal Test Accuracy: %.2f%%\n", finalTestAcc);
//    printf("Final Test Loss: %.4f\n", finalTestLoss);
//    printf("Total Training Time: %.1f seconds\n", totalTime);
//
//    TF_DeleteStatus(runStatus);
//
//    // ----- 정리 -----
//    TF_DeleteSession(session, status);
//    TF_DeleteSessionOptions(options);
//    TF_DeleteGraph(graph);
//    TF_DeleteStatus(status);
//
//    printf("\nDone!\n");
//
//    return 0;
//}
//