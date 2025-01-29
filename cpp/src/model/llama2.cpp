#include "model/llama2.h"
#include <iostream>
#include <iomanip>

using namespace toy::llama2;
typedef Tensor::DimType64 CastInt64;

LlamaRMSNorm::LlamaRMSNorm(const size_t start, const size_t hiddenSize, char *modelWeight, const DataType &dataType)
    : isMultiThread(false)
{
    size_t typeSize = toy::common::getTypeSize(dataType);

    Tensor::Shape weightShape = Tensor::makeShape({CastInt64(hiddenSize)});
    weight = Tensor::wrap(modelWeight + start * typeSize, dataType, weightShape, hiddenSize);
}

void LlamaRMSNorm::forward(Tensor::UniquePtr &out, Tensor::UniquePtr &inp)
{
    kernel::launch::rmsNorm(out, inp, weight, isMultiThread);
}

LlamaRotaryEmbedding::LlamaRotaryEmbedding(LlamaConfig &config)
    : isMultiThread(false), headDims(config.hiddenSize / config.numAttentionHeads), maxPos(config.maxPositionEmbeddings), theta(config.theta)
{
    freqsCosSin = func::createTensor(
        {CastInt64(config.maxPositionEmbeddings), CastInt64(2), CastInt64(config.hiddenSize / 2)},
        config.dataType, MemoryType::kCPU);
    kernel::launch::precomputeFreqsCosSin(freqsCosSin, headDims, maxPos, theta, isMultiThread);
}

void LlamaRotaryEmbedding::forward(Tensor::UniquePtr &inp, Tensor::UniquePtr &pos)
{
    kernel::launch::applyRope(inp, freqsCosSin, pos, isMultiThread);
}

AttentionSpace::AttentionSpace(LlamaConfig &config, runtimeParams &params) : config(config), params(params)
{
    queryStates = func::createTensor(
        {CastInt64(params.batch), CastInt64(params.length), CastInt64(config.hiddenSize)},
        config.dataType, MemoryType::kCPU);
    queryStatesTransposed = func::createTensor(
        {CastInt64(params.batch), CastInt64(config.numAttentionHeads), CastInt64(params.length), CastInt64(config.hiddenSize / config.numAttentionHeads)},
        config.dataType, MemoryType::kCPU);
    attentionScores = func::createTensor(
        {CastInt64(params.batch), CastInt64(config.numAttentionHeads), CastInt64(params.length), CastInt64(params.length)},
        config.dataType, MemoryType::kCPU);
    attentionOutput = func::createTensor(
        {CastInt64(params.batch), CastInt64(config.numAttentionHeads), CastInt64(params.length), CastInt64(config.hiddenSize / config.numAttentionHeads)},
        config.dataType, MemoryType::kCPU);
    attentionOutputTransposed = func::createTensor(
        {CastInt64(params.batch), CastInt64(params.length), CastInt64(config.hiddenSize)},
        config.dataType, MemoryType::kCPU);
    attentionOutputProjected = func::createTensor(
        {CastInt64(params.batch), CastInt64(params.length), CastInt64(config.hiddenSize)},
        config.dataType, MemoryType::kCPU);
    kvCache = func::createTensor(
        {CastInt64(config.layerNums), CastInt64(2), CastInt64(params.batch), CastInt64(config.maxPositionEmbeddings), CastInt64(config.hiddenSize)},
        config.dataType, MemoryType::kCPU);
    kvTransposed = func::createTensor(
        {CastInt64(2), CastInt64(params.batch), CastInt64(config.numAttentionHeads), CastInt64(config.maxPositionEmbeddings), CastInt64(config.hiddenSize / config.numAttentionHeads)},
        config.dataType, MemoryType::kCPU);
}

void AttentionSpace::transToDecode()
{
    queryStates.reset();
    queryStatesTransposed.reset();
    attentionScores.reset();
    attentionOutput.reset();
    attentionOutputTransposed.reset();
    attentionOutputProjected.reset();

    queryStates = func::createTensor(
        {CastInt64(params.batch), CastInt64(1), CastInt64(config.hiddenSize)},
        DataType::kFLOAT, MemoryType::kCPU);
    queryStatesTransposed = func::createTensor(
        {CastInt64(params.batch), CastInt64(config.numAttentionHeads), CastInt64(1), CastInt64(config.hiddenSize / config.numAttentionHeads)},
        DataType::kFLOAT, MemoryType::kCPU);
    attentionScores = func::createTensor(
        {CastInt64(params.batch), CastInt64(config.numAttentionHeads), CastInt64(1), CastInt64(config.maxPositionEmbeddings)},
        DataType::kFLOAT, MemoryType::kCPU);
    attentionOutput = func::createTensor(
        {CastInt64(params.batch), CastInt64(config.numAttentionHeads), CastInt64(1), CastInt64(config.hiddenSize / config.numAttentionHeads)},
        DataType::kFLOAT, MemoryType::kCPU);
    attentionOutputTransposed = func::createTensor(
        {CastInt64(params.batch), CastInt64(1), CastInt64(config.hiddenSize)},
        DataType::kFLOAT, MemoryType::kCPU);
    attentionOutputProjected = func::createTensor(
        {CastInt64(params.batch), CastInt64(1), CastInt64(config.hiddenSize)},
        DataType::kFLOAT, MemoryType::kCPU);
}

LlamaAttention::LlamaAttention(LlamaConfig &config, const size_t layerIdx, char *modelWeight, const size_t start)
    : hiddenSize(config.hiddenSize), numAttentionHeads(config.numAttentionHeads), headDims(config.hiddenSize / config.numAttentionHeads), maxPos(config.maxPositionEmbeddings), typeSize(config.typeSize)

{
    // attentionSpace = AttentionSpace::getInstance(config);
    Tensor::Shape weightShape = Tensor::makeShape({CastInt64(config.hiddenSize), CastInt64(config.hiddenSize)});
    qProjWeight = Tensor::wrap(modelWeight + start * config.typeSize, config.dataType, weightShape, config.hiddenSize * config.hiddenSize);
    kProjWeight = Tensor::wrap(modelWeight + (start + config.hiddenSize * config.hiddenSize) * config.typeSize, config.dataType, weightShape, config.hiddenSize * config.hiddenSize);
    vProjWeight = Tensor::wrap(modelWeight + (start + 2 * config.hiddenSize * config.hiddenSize) * config.typeSize, config.dataType, weightShape, config.hiddenSize * config.hiddenSize);
    oProjWeight = Tensor::wrap(modelWeight + (start + 3 * config.hiddenSize * config.hiddenSize) * config.typeSize, config.dataType, weightShape, config.hiddenSize * config.hiddenSize);

    // qStatesPrefill = func::createTensor()
}

void LlamaAttention::forward(Tensor::UniquePtr &hiddenStatesOut,
                             Tensor::UniquePtr &hiddenStatesIn,
                             const size_t layerIdx,
                             Tensor::UniquePtr &pos,
                             const size_t pastToken,
                             LlamaRotaryEmbedding &rotaryEmbedding,
                             AttentionSpace &attentionSpace)
{
    size_t qLen = hiddenStatesIn->getShape().d[1];
    std::cout << "qLen: " << qLen << std::endl;
    size_t bsz = hiddenStatesIn->getShape().d[0];
    bool isPrefill = true ? qLen > 1 : false;
    if (!isPrefill)
    {
        std::cout << "start transing!" << std::endl;
        attentionSpace.transToDecode();
        std::cout << "transed!" << std::endl;
    }

    // pos = func::makeRange(0, qLen, 1, MemoryType::kCPU);
    //  wrap key and value with kvCache
    Tensor::Shape kvShape = Tensor::makeShape({CastInt64(bsz), CastInt64(qLen), CastInt64(hiddenSize)});
    void *kvCacheData = attentionSpace.kvCache->data();
    size_t offsetK = layerIdx * 2 * bsz * maxPos * hiddenSize + pastToken * bsz * hiddenSize;
    size_t offsetV = layerIdx * 2 * bsz * maxPos * hiddenSize + bsz * maxPos * hiddenSize + pastToken * bsz * hiddenSize;
    Tensor::UniquePtr keyStates = Tensor::wrap(static_cast<char *>(kvCacheData) + offsetK * typeSize, DataType::kFLOAT, kvShape);
    Tensor::UniquePtr valueStates = Tensor::wrap(static_cast<char *>(kvCacheData) + offsetV * typeSize, DataType::kFLOAT, kvShape);
    kernel::launch::matmulWeight(attentionSpace.queryStates, hiddenStatesIn, qProjWeight, nullptr, kernel::cpu::MatmulType::KMatmulMultiThread);
    kernel::launch::matmulWeight(keyStates, hiddenStatesIn, kProjWeight, nullptr, kernel::cpu::MatmulType::KMatmulMultiThread);
    kernel::launch::matmulWeight(valueStates, hiddenStatesIn, vProjWeight, nullptr, kernel::cpu::MatmulType::KMatmulMultiThread);
    func::reShape(attentionSpace.queryStates, {CastInt64(bsz), CastInt64(qLen), CastInt64(numAttentionHeads), CastInt64(headDims)});
    func::reShape(keyStates, {CastInt64(bsz), CastInt64(qLen), CastInt64(numAttentionHeads), CastInt64(headDims)});
    func::reShape(valueStates, {CastInt64(bsz), CastInt64(qLen), CastInt64(numAttentionHeads), CastInt64(headDims)});
    rotaryEmbedding.forward(attentionSpace.queryStates, pos);
    rotaryEmbedding.forward(keyStates, pos);
    kernel::launch::transpose(attentionSpace.queryStatesTransposed, attentionSpace.queryStates, false);
    size_t offsetVTransposed = attentionSpace.kvTransposed->getSize() / 2;
    size_t kvLength = pastToken + qLen;
    Tensor::Shape kvTransposedShape = Tensor::makeShape({CastInt64(bsz), CastInt64(numAttentionHeads), CastInt64(kvLength), CastInt64(headDims)});
    Tensor::UniquePtr keyStatesTransposed = Tensor::wrap(attentionSpace.kvTransposed->data(), DataType::kFLOAT, kvTransposedShape);
    Tensor::UniquePtr valueStatesTransposed = Tensor::wrap(static_cast<char *>(attentionSpace.kvTransposed->data()) + offsetVTransposed * typeSize, DataType::kFLOAT, kvTransposedShape);
    if (!isPrefill)
    {
        offsetK = layerIdx * 2 * bsz * maxPos * hiddenSize;
        offsetV = offsetK + bsz * maxPos * hiddenSize;
        kvShape = Tensor::makeShape({CastInt64(bsz), CastInt64(kvLength), CastInt64(numAttentionHeads), CastInt64(headDims)});
        keyStates = Tensor::wrap(static_cast<char *>(kvCacheData) + offsetK * typeSize, DataType::kFLOAT, kvShape);
        valueStates = Tensor::wrap(static_cast<char *>(kvCacheData) + offsetV * typeSize, DataType::kFLOAT, kvShape);
    }
    kernel::launch::transpose(keyStatesTransposed, keyStates, false);
    kernel::launch::transpose(valueStatesTransposed, valueStates, false);
    Tensor::Shape attentionScoresShape = Tensor::makeShape({CastInt64(bsz), CastInt64(numAttentionHeads), CastInt64(qLen), CastInt64(kvLength)});
    Tensor::UniquePtr attentionScore = Tensor::wrap(attentionSpace.attentionScores->data(), DataType::kFLOAT, attentionScoresShape);
    kernel::launch::attentionForward(attentionSpace.attentionOutput, attentionSpace.queryStatesTransposed, keyStatesTransposed, valueStatesTransposed, attentionScore, isPrefill, kernel::cpu::AttentionType::kAttentionOneThread);
    kernel::launch::transpose(attentionSpace.attentionOutputTransposed, attentionSpace.attentionOutput, false);
    kernel::launch::matmulWeight(hiddenStatesOut, attentionSpace.attentionOutputTransposed, oProjWeight, nullptr, kernel::cpu::MatmulType::KMatmulMultiThread);
    func::reShape(hiddenStatesOut, {CastInt64(bsz), CastInt64(qLen), CastInt64(hiddenSize)});
}