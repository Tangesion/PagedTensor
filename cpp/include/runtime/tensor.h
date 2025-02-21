
#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include "common/dataType.h"
#include <numeric>
#include <stdexcept>
#include "buffer.h"
#include "common/assert.h"
#include "block.h"

namespace toy::runtime
{

    class Dims
    {
    public:
        static constexpr int32_t MAX_DIMS{8};

        int32_t nbDims;

        int64_t d[MAX_DIMS];
    };

    class Tensor : virtual public Buffer
    {
    public:
        using Shape = Dims;
        using UniquePtr = std::unique_ptr<Tensor>;
        using SharedPtr = std::shared_ptr<Tensor>;
        using UniqueConstPtr = std::unique_ptr<Tensor const>;
        using SharedConstPtr = std::shared_ptr<Tensor const>;
        using DataType = toy::common::DataType;
        using DimType64 = std::remove_reference_t<decltype(Shape::d[0])>;

        static UniquePtr wrap(void *data, DataType type, Shape const &shape, std::size_t capacity);
        static UniquePtr wrap(void *data, DataType type, Shape const &shape)
        {
            return wrap(data, type, shape, volume(shape));
        }
        Tensor &operator=(Tensor const &) = delete;
        Tensor(Tensor const &) = delete;
        ~Tensor() override = default;

        [[nodiscard]] virtual Shape const &getShape() const = 0;

        virtual void reshape(Shape const &dims) = 0;

        static std::size_t volume(Dims dim)
        {
            return std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<std::size_t>());
        }
        static std::size_t volume(Dims dim, size_t blockNum, size_t blockSize, DataType type)
        {
            BlockManager::getInstance().initialize(blockNum, blockSize, type);
            return std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<std::size_t>());
        }

        static Shape makeShape(std::initializer_list<DimType64> const &dims);

        static Shape makeShape(std::vector<int64_t> const &dims);

        void printShape() const;

    protected:
        Tensor() = default;
    };

    std::ostream &operator<<(std::ostream &output, Tensor const &tensor);
} // namespaece toy::runtime