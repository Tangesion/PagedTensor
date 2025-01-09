#pragma once
#include "include/func/func.h"
#include "include/kernel/launch/attention.h"
#include "include/kernel/launch/ffn.h"
#include "include/kernel/launch/matmul.h"
#include "include/kernel/launch/rope.h"
#include "include/kernel/launch/rmsnorm.h"

using SharedPtr = inference_frame::runtime::Tensor::SharedPtr;

namespace inference_frame::llama2
{
    class LlamaConfig
    {
    public:
        LlamaConfig(
            size_t vocabSize,
            size_t hiddenSize,
            size_t intermediateSize,
            size_t numHiddenLayers,
            size_t numAttentionHeads,
            size_t maxPositionEmbeddings)
            : vocabSize(vocabSize),
              hiddenSize(hiddenSize),
              intermediateSize(intermediateSize),
              numHiddenLayers(numHiddenLayers),
              numAttentionHeads(numAttentionHeads),
              maxPositionEmbeddings(maxPositionEmbeddings)
        {
        }
        ~LlamaConfig() = default;
        size_t vocabSize;
        size_t hiddenSize;
        size_t intermediateSize;
        size_t numHiddenLayers;
        size_t numAttentionHeads;
        size_t maxPositionEmbeddings;
    };

    class LlamaRMSNorm
    {
    public:
        LlamaRMSNorm(size_t start, size_t hiddenSize, SharedPtr const &modelWeight);

    private:
        bool isMultiThread;
        SharedPtr weight;
    };
}
