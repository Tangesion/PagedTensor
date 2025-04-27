#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <iostream>
#include "common/dataPtr.h"
#include "common/dataType.h"
using DataPtr = paged_tensor::common::DataPtr;
namespace paged_tensor::runtime
{
    class Block;
    // class DataPtr;
    enum class MemoryType : std::int32_t
    {
        kGPU = 0,
        kCPU = 1,
        kPINNED = 2,
        kUVM = 3,
        kPINNEDPOOL = 4
    };

    using DataType = paged_tensor::common::DataType;

    class BufferDataType
    {

    public:
        constexpr BufferDataType(DataType dataType) : m_dataType(dataType)
        {
        }

        [[nodiscard]] constexpr DataType getDataType() const
        {
            return m_dataType;
        }

        [[nodiscard]] constexpr std::size_t getSize() const noexcept
        {
            return paged_tensor::common::getTypeSize(m_dataType);
        }

    private:
        DataType m_dataType;
    };

    class Buffer
    {

    public:
        using DataType = paged_tensor::common::DataType;
        using SharedPtr = std::shared_ptr<Buffer>;
        using UniquePtr = std::unique_ptr<Buffer>;

        // virtual void generateRandomData() = 0;

        [[nodiscard]] virtual DataPtr dataPaged() const = 0;

        [[nodiscard]] virtual void *data() = 0;

        [[nodiscard]] virtual void const *data() const = 0;

        [[nodiscard]] virtual void *data(std::size_t index)
        {
            auto *const dataPtr = this->data();
            return dataPtr ? static_cast<std::uint8_t *>(dataPtr) + toBytes(index) : nullptr;
        }
        [[nodiscard]] virtual DataType getDataType() const = 0;

        [[nodiscard]] virtual MemoryType getMemoryType() const = 0;

        static MemoryType memoryType();

        [[nodiscard]] virtual std::size_t getSize() const = 0;

        [[nodiscard]] virtual std::size_t getCapacity() const = 0;

        virtual void release() = 0;

        virtual void resize(std::size_t size) = 0;

        virtual std::vector<void *> *getBlockMap() = 0;

        [[nodiscard]] virtual bool isPaged() const = 0;

        virtual ~Buffer() = default;

    protected:
        Buffer() = default;

        [[nodiscard]] std::size_t toBytes(std::size_t size) const
        {
            return size * BufferDataType(getDataType()).getSize();
        }
    };

} // namespace paged_tensor::runtime