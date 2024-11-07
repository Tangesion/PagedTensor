#include "runtime/Tensor.h"
#include <iostream>

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
        // tensor.reset( // NOLINT(modernize-make-unique)
        //     new GenericTensor<CpuBorrowingAllocator>(
        //         shape, capacity, type, CpuBorrowingAllocator(data, capacityInBytes)));
        break;

    default:
        break;
    }
}

Tensor::Shape Tensor::makeShape(std::initializer_list<Tensor::DimType64> const &dims)
{
    Dims shape{};
    shape.nbDims = static_cast<decltype(Shape::nbDims)>(dims.size());
    std::copy(dims.begin(), dims.end(), shape.d);
    return shape;
}

void Tensor::printShape()
{
    Shape shape = this->getShape();
    std::string str = "shape [";
    for (int i = 0; i < shape.nbDims; i++)
    {
        str += std::to_string(shape.d[i]);
        if (i < shape.nbDims - 1)
        {
            str += ", ";
        }
    }
    str += "]";
    std::cout << str << std::endl;
}
