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
class AttentionTest
{
public:
    AttentionTest(LlamaConfig &config, runtimeParams &params, std::string modelPath);

    torch::Tensor getQProj(LlamaConfig &config);
    torch::Tensor getKProj(LlamaConfig &config);
    torch::Tensor getVProj(LlamaConfig &config);
    torch::Tensor getOProj(LlamaConfig &config);

    torch::Tensor forwardTest(torch::Tensor &output,
                              torch::Tensor &input,
                              const size_t layerIdx,
                              torch::Tensor &pos,
                              const size_t pastToken)
    {
        Tensor::UniquePtr outputpaged_tensor = paged_tensor::utils::torchTopaged_tensor(output);
        Tensor::UniquePtr inputpaged_tensor = paged_tensor::utils::torchTopaged_tensor(input);
        Tensor::UniquePtr pospaged_tensor = paged_tensor::utils::torchTopaged_tensor(pos);
        // std::cout << "ready" << std::endl;
        attention.forward(outputpaged_tensor, inputpaged_tensor, layerIdx, pospaged_tensor, pastToken, rotaryEmbedding, attentionSpace);
        // std::cout << *outputpaged_tensor << std::endl;
        return output;
    }

    void printMessage()
    {
        std::cout << "Bind AttentionTest successed!" << std::endl;
    }

private:
    std::vector<char> modelWeight;
    size_t start;
    LlamaAttention attention;
    LlamaRotaryEmbedding rotaryEmbedding;
    AttentionSpace attentionSpace;
};

AttentionTest::AttentionTest(LlamaConfig &config, runtimeParams &params, std::string modelPath)
    : start(config.vocabSize * config.hiddenSize), attention(config, 0, nullptr, start), rotaryEmbedding(config), attentionSpace(config, params)
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

    attention = LlamaAttention(config, 0, modelWeight.data(), start);
}

torch::Tensor AttentionTest::getQProj(LlamaConfig &config)
{
    void *qProjdata = attention.getQProjWeight()->data();
    torch::Tensor qProj = torch::from_blob(qProjdata, {static_cast<int64_t>(config.hiddenSize), static_cast<int64_t>(config.hiddenSize)}, torch::kFloat);
    return qProj;
}
torch::Tensor AttentionTest::getKProj(LlamaConfig &config)
{
    void *kProjdata = attention.getKProjWeight()->data();
    torch::Tensor kProj = torch::from_blob(kProjdata, {static_cast<int64_t>(config.hiddenSize), static_cast<int64_t>(config.hiddenSize)}, torch::kFloat);
    return kProj;
}
torch::Tensor AttentionTest::getVProj(LlamaConfig &config)
{
    void *vProjdata = attention.getVProjWeight()->data();
    torch::Tensor vProj = torch::from_blob(vProjdata, {static_cast<int64_t>(config.hiddenSize), static_cast<int64_t>(config.hiddenSize)}, torch::kFloat);
    return vProj;
}
torch::Tensor AttentionTest::getOProj(LlamaConfig &config)
{
    void *oProjdata = attention.getOProjWeight()->data();
    torch::Tensor oProj = torch::from_blob(oProjdata, {static_cast<int64_t>(config.hiddenSize), static_cast<int64_t>(config.hiddenSize)}, torch::kFloat);
    return oProj;
}

PYBIND11_MODULE(attentionClass, m)
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

    pybind11::class_<runtimeParams>(m, "runtimeParams")
        .def(pybind11::init<size_t, size_t>())
        .def_readwrite("batch", &runtimeParams::batch)
        .def_readwrite("length", &runtimeParams::length);

    pybind11::class_<AttentionTest>(m, "AttentionTest")
        .def(pybind11::init<LlamaConfig &, runtimeParams &, std::string>())
        .def("printMessage", &AttentionTest::printMessage)
        .def("getQProj", &AttentionTest::getQProj)
        .def("getKProj", &AttentionTest::getKProj)
        .def("getVProj", &AttentionTest::getVProj)
        .def("getOProj", &AttentionTest::getOProj)
        .def("forwardTest", &AttentionTest::forwardTest);
}