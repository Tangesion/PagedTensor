#pragma once
#include "common/assert.h"
#include "func/func.h"
#include "kernel/launch/attention.h"
#include "kernel/launch/ffn.h"
#include "kernel/launch/matmul.h"
#include "kernel/launch/rope.h"
#include "kernel/launch/rmsnorm.h"
#include "kernel/launch/transpose.h"

using namespace paged_tensor::runtime;

namespace paged_tensor::llama2
{

    class WorkSpace
    {
    public:
        WorkSpace();
    };

    struct runtimeParams
    {
        size_t batch;
        size_t length;
        runtimeParams(size_t batch, size_t length) : batch(batch), length(length)
        {
            try
            {
                CHECK_WITH_INFO(batch == 1, "now we only support batch size 1");
                CHECK_WITH_INFO(length > 0, "length must be greater than 0");
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
                std::exit(EXIT_FAILURE);
            }
        }
    };

    class LlamaConfig
    {
    public:
        LlamaConfig(
            size_t vocabSize,
            size_t hiddenSize,
            size_t intermediateSize,
            size_t numHiddenLayers,
            size_t numAttentionHeads,
            size_t maxPositionEmbeddings,
            size_t layerNums,
            float theta,
            Tensor::DataType dataType,
            size_t invFreqSize = 64)
            : vocabSize(vocabSize),
              hiddenSize(hiddenSize),
              intermediateSize(intermediateSize),
              numHiddenLayers(numHiddenLayers),
              numAttentionHeads(numAttentionHeads),
              maxPositionEmbeddings(maxPositionEmbeddings),
              layerNums(layerNums),
              invFreqSize(invFreqSize),
              theta(theta),
              dataType(dataType)
        {
            try
            {
                CHECK_WITH_INFO(hiddenSize % numAttentionHeads == 0, "hiddenSize must be divisible by numAttentionHeads");
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
                std::exit(EXIT_FAILURE);
            }
            typeSize = paged_tensor::common::getTypeSize(dataType);
        }
        ~LlamaConfig() = default;
        size_t vocabSize;
        size_t hiddenSize;
        size_t intermediateSize;
        size_t numHiddenLayers;
        size_t numAttentionHeads;
        size_t maxPositionEmbeddings;
        size_t layerNums;
        size_t invFreqSize;
        float theta;
        Tensor::DataType dataType;
        size_t typeSize;
    };

    class LlamaRMSNorm
    {
    public:
        LlamaRMSNorm(const size_t start, const size_t hiddenSize, char *modelWeight, const DataType &dataType);
        void forward(Tensor::UniquePtr &out, Tensor::UniquePtr &inp);

    private:
        bool isMultiThread;
        Tensor::UniquePtr weight;
    };

    class LlamaRotaryEmbedding
    {
    public:
        LlamaRotaryEmbedding(LlamaConfig &config);
        void forward(Tensor::UniquePtr &inp, Tensor::UniquePtr &pos);

    private:
        bool isMultiThread;
        size_t headDims;
        size_t maxPos;
        float theta;
        Tensor::UniquePtr freqsCosSin;
    };

    class AttentionSpace
    {
        // public:
        //     static AttentionSpace &getInstance(LlamaConfig &config);

    public:
        AttentionSpace(LlamaConfig &config, runtimeParams &params);
        void transToDecode();
        void printMemory();

    public:
        Tensor::UniquePtr queryStates;
        Tensor::UniquePtr queryStatesTransposed;
        Tensor::UniquePtr attentionScores;
        Tensor::UniquePtr attentionOutput;
        Tensor::UniquePtr attentionOutputTransposed;
        Tensor::UniquePtr attentionOutputProjected;
        Tensor::UniquePtr kvCache;
        Tensor::UniquePtr kvTransposed;

    private:
        LlamaConfig config;
        runtimeParams params;

        // private:

        // AttentionSpace(AttentionSpace const &) = delete;
        // AttentionSpace &operator=(AttentionSpace const &) = delete;
    };

    class LlamaAttention
    {
    public:
        LlamaAttention(LlamaConfig &config, const size_t layerIdx, char *modelWeight, const size_t start);
        void forward(Tensor::UniquePtr &hiddenStatesOut,
                     Tensor::UniquePtr &hiddenStatesIn,
                     const size_t layerIdx,
                     Tensor::UniquePtr &pos,
                     const size_t pastToken,
                     LlamaRotaryEmbedding &rotaryEmbedding,
                     AttentionSpace &attentionSpace);

    private:
        size_t hiddenSize;
        size_t numAttentionHeads;
        size_t headDims;
        size_t maxPos;
        size_t typeSize;
        Tensor::UniquePtr qProjWeight;
        Tensor::UniquePtr kProjWeight;
        Tensor::UniquePtr vProjWeight;
        Tensor::UniquePtr oProjWeight;

    public:
        Tensor::UniquePtr &getQProjWeight() { return qProjWeight; }
        Tensor::UniquePtr &getKProjWeight() { return kProjWeight; }
        Tensor::UniquePtr &getVProjWeight() { return vProjWeight; }
        Tensor::UniquePtr &getOProjWeight() { return oProjWeight; }
    };

    class LlamaPagedAttention
    {
    public:
        LlamaPagedAttention(LlamaConfig &config, const size_t layerIdx, char *modelWeight, const size_t start);
    };

    class LlamaMLP
    {
    public:
        LlamaMLP(LlamaConfig &config, char *modelWeight, const size_t start);
        void forward(Tensor::UniquePtr &hiddenStatesOut, Tensor::UniquePtr &hiddenStatesIn);

    private:
        size_t hiddenSize;
        size_t intermediateSize;
        size_t typeSize;
        Tensor::UniquePtr gateProjWeight;
        Tensor::UniquePtr upProjWeight;
        Tensor::UniquePtr downProjWeight;

    public:
        Tensor::UniquePtr &getGateProjWeight() { return gateProjWeight; }
        Tensor::UniquePtr &getUpProjWeight() { return upProjWeight; }
        Tensor::UniquePtr &getDownProjWeight() { return downProjWeight; }
    };
}
