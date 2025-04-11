#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "model/llama2.h"
#include "../utils/utlis.h"

using namespace paged_tensor::llama2;

// inline findMlpStart(size_t layerIdx, LlamaConfig &config)

class MLPTest
{
public:
    MLPTest(LlamaConfig &config, std::string modelPath, size_t layerIdx);

    torch::Tensor getGateProj(LlamaConfig &config)
    {
        void *gateProjData = mlp.getGateProjWeight()->data();
        torch::Tensor gateProj = torch::from_blob(gateProjData, {static_cast<int64_t>(config.intermediateSize), static_cast<int64_t>(config.hiddenSize)}, torch::kFloat);
        return gateProj;
    }
    torch::Tensor getUpProj(LlamaConfig &config)
    {
        void *upProjData = mlp.getUpProjWeight()->data();
        torch::Tensor upProj = torch::from_blob(upProjData, {static_cast<int64_t>(config.intermediateSize), static_cast<int64_t>(config.hiddenSize)}, torch::kFloat);
        return upProj;
    }
    torch::Tensor getDownProj(LlamaConfig &config)
    {
        void *downProjData = mlp.getDownProjWeight()->data();
        torch::Tensor downProj = torch::from_blob(downProjData, {static_cast<int64_t>(config.hiddenSize), static_cast<int64_t>(config.intermediateSize)}, torch::kFloat);
        return downProj;
    }

    size_t getStart(LlamaConfig &config, size_t layerIdx)
    {
        size_t embedTokenSize = config.vocabSize * config.hiddenSize;
        size_t attentionProjSize = config.hiddenSize * config.hiddenSize;
        size_t rotatyEmbSize = config.invFreqSize;
        size_t mlpProjSize = config.hiddenSize * config.intermediateSize;
        size_t layerNormSize = config.hiddenSize;
        size_t oneBlockSize = attentionProjSize * 4 + rotatyEmbSize + mlpProjSize * 3 + layerNormSize * 2;

        return embedTokenSize + oneBlockSize * layerIdx + attentionProjSize * 4 + rotatyEmbSize;
    }

    torch::Tensor forwardTest(torch::Tensor &output, torch::Tensor &input)
    {
        Tensor::UniquePtr outputPagedTensor = paged_tensor::utils::torchToPagedTensor(output);
        Tensor::UniquePtr inputPagedTensor = paged_tensor::utils::torchToPagedTensor(input);
        mlp.forward(outputPagedTensor, inputPagedTensor);
        return output;
    }

    void printMessage()
    {
        std::cout << "bind mlp successfully!" << std::endl;
    }

private:
    LlamaMLP mlp;
    std::vector<char> modelWeight;
    // size_t start;
};

MLPTest::MLPTest(LlamaConfig &config, std::string modelPath, size_t layerIdx)
    : mlp(config, nullptr, 0)
{
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "can't open!: " << modelPath << std::endl;
        return;
    }
    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize <= 0)
    {
        std::cerr << "Invalid file size: " << fileSize << std::endl;
        return;
    }

    std::cout << "fileSize(MB): " << std::fixed << std::setprecision(2) << static_cast<double>(fileSize) / (1024 * 1024) << std::endl;
    std::vector<char> buffer(fileSize);
    modelWeight.resize(fileSize);
    if (!file.read(modelWeight.data(), fileSize))
    {
        std::cerr << "read fail!: " << modelPath << std::endl;
        return;
    }
    std::cout << "File read successfully" << std::endl;
    size_t start = getStart(config, layerIdx);

    mlp = LlamaMLP(config, modelWeight.data(), start);
}

PYBIND11_MODULE(mlp_class, m)
{
    pybind11::enum_<paged_tensor::common::DataType>(m, "DataType")
        .value("FLOAT32", paged_tensor::common::DataType::kFLOAT)
        .export_values();

    pybind11::class_<LlamaConfig>(m, "LlamaConfig")
        .def(pybind11::init<size_t, size_t, size_t, size_t, size_t, size_t, size_t, float, Tensor::DataType>())
        .def_readwrite("vocabSize", &LlamaConfig::vocabSize)
        .def_readwrite("hiddenSize", &LlamaConfig::hiddenSize)
        .def_readwrite("intermediateSize", &LlamaConfig::intermediateSize)
        .def_readwrite("numHiddenLayers", &LlamaConfig::numHiddenLayers)
        .def_readwrite("numAttentionHeads", &LlamaConfig::numAttentionHeads)
        .def_readwrite("maxPositionEmbeddings", &LlamaConfig::maxPositionEmbeddings)
        .def_readwrite("layerNums", &LlamaConfig::layerNums)
        .def_readwrite("theta", &LlamaConfig::theta)
        .def_readwrite("dataType", &LlamaConfig::dataType)
        .def_readwrite("typeSize", &LlamaConfig::typeSize);

    pybind11::class_<MLPTest>(m, "MLPTest")
        .def(pybind11::init<LlamaConfig &, std::string, size_t>())
        .def("printMessage", &MLPTest::printMessage)
        .def("getGateProj", &MLPTest::getGateProj)
        .def("getUpProj", &MLPTest::getUpProj)
        .def("getDownProj", &MLPTest::getDownProj)
        .def("forwardTest", &MLPTest::forwardTest);
}