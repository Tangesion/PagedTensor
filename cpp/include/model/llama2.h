#pragma once
#include "include/common/assert.h"
#include "include/func/func.h"
#include "include/kernel/launch/attention.h"
#include "include/kernel/launch/ffn.h"
#include "include/kernel/launch/matmul.h"
#include "include/kernel/launch/rope.h"
#include "include/kernel/launch/rmsnorm.h"
#include "include/kernel/launch/transpose.h"

using namespace inference_frame::runtime;

namespace inference_frame::llama2
{

    class WorkSpace
    {
    public:
        WorkSpace();
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
            size_t batch,
            size_t prefillLength,
            size_t layerNums,
            float theta,
            Tensor::DataType dataType)
            : vocabSize(vocabSize),
              hiddenSize(hiddenSize),
              intermediateSize(intermediateSize),
              numHiddenLayers(numHiddenLayers),
              numAttentionHeads(numAttentionHeads),
              maxPositionEmbeddings(maxPositionEmbeddings),
              batch(batch),
              prefillLength(prefillLength),
              layerNums(layerNums),
              theta(theta),
              dataType(dataType)
        {
            try
            {
                CHECK_WITH_INFO(batch == 1, "Now we only support batch size 1");
                CHECK_WITH_INFO(hiddenSize % numAttentionHeads == 0, "hiddenSize must be divisible by numAttentionHeads");
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
                std::exit(EXIT_FAILURE);
            }
            typeSize = inference_frame::common::getTypeSize(dataType);
        }
        ~LlamaConfig() = default;
        size_t vocabSize;
        size_t hiddenSize;
        size_t intermediateSize;
        size_t numHiddenLayers;
        size_t numAttentionHeads;
        size_t maxPositionEmbeddings;
        size_t batch;
        size_t prefillLength;
        size_t layerNums;
        float theta;
        Tensor::DataType dataType;
        size_t typeSize;
    };

    class LlamaRMSNorm
    {
    public:
        LlamaRMSNorm(const size_t start, const size_t hiddenSize, void *modelWeight, const DataType &dataType);
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
        AttentionSpace(LlamaConfig &config);
        void transToDecode(LlamaConfig &config);

    public:
        Tensor::UniquePtr queryStates;
        Tensor::UniquePtr queryStatesTransposed;
        Tensor::UniquePtr keyStatesTransposed;
        Tensor::UniquePtr valueStatesTransposed;
        Tensor::UniquePtr attentionScores;
        Tensor::UniquePtr attentionOutput;
        Tensor::UniquePtr attentionOutputTransposed;
        Tensor::UniquePtr attentionOutputProjected;
        Tensor::UniquePtr kvCache;

        // private:

        // AttentionSpace(AttentionSpace const &) = delete;
        // AttentionSpace &operator=(AttentionSpace const &) = delete;
    };

    class LlamaAttention
    {
    public:
        LlamaAttention(LlamaConfig &config, const size_t layerIdx, void *modelWeight, const size_t start);
        void forward(Tensor::UniquePtr hiddenStatesOut,
                     Tensor::UniquePtr hiddenStatesIn,
                     const size_t layerIdx,
                     LlamaRotaryEmbedding &rotaryEmbedding,
                     AttentionSpace &attentionSpace);

    private:
        size_t hiddenSize;
        size_t numAttentionHeads;
        size_t headDims;
        size_t prefillLength;
        size_t maxPos;
        size_t typeSize;
        Tensor::UniquePtr qProjWeight;
        Tensor::UniquePtr kProjWeight;
        Tensor::UniquePtr vProjWeight;
        Tensor::UniquePtr oProjWeight;
    };
}
