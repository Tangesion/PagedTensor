#include "runtime/tensor.h"
#include "runtime/llmBuffer.h"
#include <iostream>

using namespace paged_tensor::runtime;

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
    return tensor;
}

Tensor::Shape Tensor::makeShape(std::initializer_list<Tensor::DimType64> const &dims)
{
    Dims shape{};
    shape.nbDims = static_cast<decltype(Shape::nbDims)>(dims.size());
    std::copy(dims.begin(), dims.end(), shape.d);
    return shape;
}

Tensor::Shape Tensor::makeShape(std::vector<int64_t> const &dims)
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
    const T *printRecursive(const T *arr, const int32_t rank, const int64_t *lengths, const int32_t staticRank, std::ostream &out)
    {
        std::string p_sep = "";
        out << "[";
        if (rank > 1)
        {
            for (int64_t i = 0; i < lengths[0]; i++)
            {
                out << p_sep;
                p_sep = "";
                arr = printRecursive(arr, rank - 1, lengths + 1, staticRank, out);
                for (int32_t j = 1; j < rank; j++)
                {
                    p_sep += "\n";
                }
                for (int32_t j = 0; j <= staticRank - rank; j++)
                {
                    p_sep += " ";
                }
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
        // out << std::endl;
        return arr;
    }
    template <typename T>
    DataPtr printRecursive(DataPtr arr, const int32_t rank, const int64_t *lengths, const int32_t staticRank, std::ostream &out)
    {
        std::string p_sep = "";
        out << "[";
        if (rank > 1)
        {
            for (int64_t i = 0; i < lengths[0]; i++)
            {
                out << p_sep;
                p_sep = "";
                arr = printRecursive<T>(arr, rank - 1, lengths + 1, staticRank, out);
                for (int32_t j = 1; j < rank; j++)
                {
                    p_sep += "\n";
                }
                for (int32_t j = 0; j <= staticRank - rank; j++)
                {
                    p_sep += " ";
                }
            }
        }
        else
        {
            for (int64_t i = 0; i < lengths[0]; i++)
            {
                out << p_sep << *static_cast<T *>(arr.data());
                arr++;
                p_sep = ", ";
            }
        }
        out << "]";
        // out << std::endl;
        return arr;
    }

    template <typename T>
    void printTensor(Tensor const &tensor, std::ostream &out)
    {
        tensor.printShape();
        out << "values: " << std::endl;
        if (!tensor.isPaged())
        {
            T const *data = static_cast<T const *>(tensor.data());
            printRecursive(data, tensor.getShape().nbDims, tensor.getShape().d, tensor.getShape().nbDims, out);
        }
        else
        {
            DataPtr arr = tensor.dataPaged();
            printRecursive<T>(arr, tensor.getShape().nbDims, tensor.getShape().d, tensor.getShape().nbDims, out);
        }
    }

}

std::ostream &paged_tensor::runtime::operator<<(std::ostream &output, Tensor const &tensor)
{
    switch (tensor.getDataType())
    {
    case DataType::kFLOAT:
        printTensor<float>(tensor, output);
        break;
    case DataType::kINT64:
        printTensor<int64_t>(tensor, output);
        break;
    default:
        break;
    }
    return output;
}
