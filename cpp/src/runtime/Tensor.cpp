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

void Tensor::printShape() const
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

namespace
{
    template <typename T>
    const T *printRecursive(const T *arr, const int32_t rank, const int64_t *lengths, std::ostream &out)
    {
        const char *p_sep = "";
        out << "[";
        if (rank > 1)
        {
            for (int64_t i = 0; i < lengths[0]; i++)
            {
                out << p_sep;
                arr = printRecursive(arr, rank - 1, lengths + 1, out);
                p_sep = "\n";
            }
        }
        else
        {
            for (int64_t i = 0; i < lengths[0]; i++)
            {
                out << p_sep << *arr++;
                p_sep = ", ";
            }
        }
        out << "]";
        return arr;
    }

    template <typename T>
    void printTensor(Tensor const &tensor, std::ostream &out)
    {
        tensor.printShape();
        out << "values: " << std::endl;

        T const *data = static_cast<T const *>(tensor.data());
        printRecursive(data, tensor.getShape().nbDims, tensor.getShape().d, out);
    }

}

std::ostream &inference_frame::runtime::operator<<(std::ostream &output, Tensor const &tensor)
{
    switch (tensor.getDataType())
    {
    case DataType::kFLOAT:
        printTensor<float>(tensor, output);
        break;
    default:
        break;
    }
    return output;
}
