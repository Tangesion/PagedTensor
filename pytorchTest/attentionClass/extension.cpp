#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cstddef>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include "model/llama2.h"

using namespace inference_frame::llama2;
class AttentionTest
{
public:
    AttentionTest(LlamaConfig &config, std::string modelPath);

    torch::Tensor getQProj(LlamaConfig &config);
    torch::Tensor getKProj(LlamaConfig &config);
    torch::Tensor getVProj(LlamaConfig &config);
    torch::Tensor getOProj(LlamaConfig &config);

    void printMessage()
    {
        std::cout << "Bind AttentionTest successed!" << std::endl;
    }

private:
    char *modelWeight;
    size_t start;
    LlamaAttention attention;
};

AttentionTest::AttentionTest(LlamaConfig &config, std::string modelPath)
    : start(config.vocabSize), attention(config, 0, modelWeight, start)
{
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "can't open!: " << modelPath << std::endl;
        return;
    }

    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(fileSize);
    if (!file.read(buffer.data(), fileSize))
    {
        std::cerr << "read fail!: " << modelPath << std::endl;
        return;
    }

    modelWeight = buffer.data();
}

torch::Tensor AttentionTest::getQProj(LlamaConfig &config)
{
    void *qProjdata = modelWeight + start * config.typeSize;
    torch::Tensor qProj = torch::from_blob(qProjdata, {static_cast<int64_t>(config.hiddenSize), static_cast<int64_t>(config.hiddenSize)}, torch::kFloat);
    return qProj;
}
torch::Tensor AttentionTest::getKProj(LlamaConfig &config)
{
    void *kProjdata = modelWeight + (start + config.hiddenSize * config.hiddenSize) * config.typeSize;
    torch::Tensor kProj = torch::from_blob(kProjdata, {static_cast<int64_t>(config.hiddenSize), static_cast<int64_t>(config.hiddenSize)}, torch::kFloat);
    return kProj;
}
torch::Tensor AttentionTest::getVProj(LlamaConfig &config)
{
    void *vProjdata = modelWeight + (start + 2 * config.hiddenSize * config.hiddenSize) * config.typeSize;
    torch::Tensor vProj = torch::from_blob(vProjdata, {static_cast<int64_t>(config.hiddenSize), static_cast<int64_t>(config.hiddenSize)}, torch::kFloat);
    return vProj;
}
torch::Tensor AttentionTest::getOProj(LlamaConfig &config)
{
    void *oProjdata = modelWeight + (start + 3 * config.hiddenSize * config.hiddenSize) * config.typeSize;
    torch::Tensor oProj = torch::from_blob(oProjdata, {static_cast<int64_t>(config.hiddenSize), static_cast<int64_t>(config.hiddenSize)}, torch::kFloat);
    return oProj;
}

PYBIND11_MODULE(attentionClass, m)
{
    pybind11::enum_<inference_frame::common::DataType>(m, "DataType")
        .value("FLOAT32", inference_frame::common::DataType::kFLOAT)
        .export_values();

    pybind11::class_<LlamaConfig>(m, "LlamaConfig")
        .def(pybind11::init<size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, float, Tensor::DataType>())
        .def_readwrite("vocabSize", &LlamaConfig::vocabSize)
        .def_readwrite("hiddenSize", &LlamaConfig::hiddenSize)
        .def_readwrite("intermediateSize", &LlamaConfig::intermediateSize)
        .def_readwrite("numHiddenLayers", &LlamaConfig::numHiddenLayers)
        .def_readwrite("numAttentionHeads", &LlamaConfig::numAttentionHeads)
        .def_readwrite("maxPositionEmbeddings", &LlamaConfig::maxPositionEmbeddings)
        .def_readwrite("batch", &LlamaConfig::batch)
        .def_readwrite("prefillLength", &LlamaConfig::prefillLength)
        .def_readwrite("layerNums", &LlamaConfig::layerNums)
        .def_readwrite("theta", &LlamaConfig::theta)
        .def_readwrite("dataType", &LlamaConfig::dataType)
        .def_readwrite("typeSize", &LlamaConfig::typeSize);

    pybind11::class_<AttentionTest>(m, "AttentionTest")
        .def(pybind11::init<LlamaConfig &, std::string>())
        .def("printMessage", &AttentionTest::printMessage)
        .def("getQProj", &AttentionTest::getQProj)
        .def("getKProj", &AttentionTest::getKProj)
        .def("getVProj", &AttentionTest::getVProj)
        .def("getOProj", &AttentionTest::getOProj);
}