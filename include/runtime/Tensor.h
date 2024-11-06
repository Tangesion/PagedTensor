
#pragma once

#include <cstdint>
#include <memory>
#include <common/DataType.h>
#include <numeric>
#include <stdexcept>
#include <Buffer.h>

namespace inference_frame::runtime
{

    class Dims
    {
    public:
        Dims(int32_t nbDims) : nbDims(nbDims)
        {
            if (nbDims < 0)
            {
                throw std::invalid_argument("nbDims cannot be less than 0");
            }
        }

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
        using DataType = inference_frame::common::DataType;

        [[nodiscard]] virtual Shape const &get_shape() const = 0;

        static UniquePtr wrap(void *data, DataType type, Shape const &shape, std::size_t capacity);

        Tensor &operator=(Tensor const &) = delete;
        Tensor(Tensor const &) = delete;

        static std::size_t volume(Dims dim)
        {
            return std::accumulate(dim.d, dim.d + dim.nbDims, 1, std::multiplies<std::size_t>());
        }
    };

    class TensorView : virtual public Tensor
    {
    public:
    };

} // namespaece inference_frame::runtime