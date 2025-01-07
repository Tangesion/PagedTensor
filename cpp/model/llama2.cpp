#include "include/func/func.h"
#include "include/kernel/launch/attention.h"
#include "include/kernel/launch/ffn.h"
#include "include/kernel/launch/matmul.h"
#include "include/kernel/launch/rope.h"
#include "include/kernel/launch/rmsnorm.h"

namespace inference_frame::llama2
{
    class Attention
    {
    public:
        Attention();
    };
}