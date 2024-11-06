#include <include/runtime/Tensor.h>

using namespace inference_frame::runtime;

Tensor::UniquePtr Tensor::wrap(void *data, DataType type, Shape const &shape, std::size_t capacity)
{
    auto const size = volume(shape);
    if (size > capacity)
    {
        throw std::invalid_argument("capacity is less than the size of the tensor");
    }
    auto memoryType = Buffer::memoryType(data);

    Tensor::UniquePtr tensor;

    auto const capacityInBytes = capacity * BufferDataType(type).getSize();
    switch (memoryType)
    {
    case MemoryType::kCPU:
        tensor.reset( // NOLINT(modernize-make-unique)
            new GenericTensor<CpuBorrowingAllocator>(
                shape, capacity, type, CpuBorrowingAllocator(data, capacityInBytes)));
        break;

    default:
        break;
    }
}
