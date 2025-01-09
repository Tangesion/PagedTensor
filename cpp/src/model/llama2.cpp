#include "include/model/llama2.h"

using namespace inference_frame::llama2;

LlamaRMSNorm::LlamaRMSNorm(size_t start, size_t hiddenSize, SharedPtr const &modelWeight)
    : isMultiThread(false)
{
}