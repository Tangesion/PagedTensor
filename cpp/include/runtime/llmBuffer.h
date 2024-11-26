#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "Buffer.h"
#include "common/DataType.h"
#include "Tensor.h"
#include "common/assert.h"

namespace inference_frame::runtime
{
    template <typename TDerived, MemoryType memoryType, bool count = false>
    class BaseAllocator
    {
    public:
        static auto constexpr kMemoryType = memoryType;

        void *allocate(std::size_t n)
        {
            void *ptr{};
            static_cast<TDerived *>(this)->allocateImpl(&ptr, n);
            if constexpr (count)
            {
                // MemoryCounters::getInstance().allocate<memoryType>(n);
            }
            return ptr;
        }

        void *deallocate(void *ptr, std::size_t n)
        {
            if (ptr)
            {
                static_cast<TDerived *>(this)->deallocateImpl(ptr, n);
                if constexpr (count)
                {
                    // MemoryCounters::getInstance().deallocate<memoryType>(n);
                }
            }
        }
        [[nodiscard]] MemoryType constexpr getMemoryType() const
        {
            return memoryType;
        }
    };

    class HostAllocator : public BaseAllocator<HostAllocator, MemoryType::kCPU, false>
    {
        friend class BaseAllocator<HostAllocator, MemoryType::kCPU, false>;

    public:
        HostAllocator() noexcept = default;

    protected:
        void allocateImpl(void **ptr, std::size_t n) // NOLINT(readability-convert-member-functions-to-static)
        {
            *ptr = std::malloc(n);
            if (!*ptr)
            {
                throw std::bad_alloc();
            }
        }

        void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
            void *ptr, [[maybe_unused]] std::size_t n)
        {
            std::free(ptr);
        }
    };

    template <MemoryType memoryType>
    class BorrowingAllocator : public BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false>
    {
        friend class BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false>;

    public:
        using Base = BaseAllocator<BorrowingAllocator<memoryType>, memoryType, false>;

        BorrowingAllocator(void *ptr, std::size_t capacity)
            : mPtr(ptr), mCapacity(capacity)
        {
            CHECK_WITH_INFO(capacity == 0 || ptr, "ptr must be non-null if capacity is non-zero");
            CHECK_WITH_INFO(mCapacity >= std::size_t{0}, "capacity must be non-negative");
        }

    protected:
        void allocateImpl(void *ptr, std::size_t n) // NOLINT(readability-convert-member-functions-to-static)
        {
            if (n <= mCapacity)
            {
                *ptr = mPtr;
            }
            else
            {
                throw std::bad_alloc();
            }
        }

        void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
            [[maybe_unused]] void *ptr, [[maybe_unused]] std::size_t n)
        {
        }

    private:
        void *mPtr;
        std::size_t mCapacity;
    };

    using CpuBorrowingAllocator = BorrowingAllocator<MemoryType::kCPU>;

    template <typename TAllocator>
    class GenericBuffer : virtual public Buffer
    {
    public:
        using AllocatorType = TAllocator;

        // Construct an empty buffer
        explicit GenericBuffer(
            DataType type, TAllocator allocator = {})
            : GenericBuffer{0, type, std::move(allocator)} {};

        // Construct a buffer with a given size

        explicit GenericBuffer(
            std::size_t size, DataType type, TAllocator allocator = {})
            : GenericBuffer{size, size, type, std::move(allocator)} {};

        GenericBuffer(GenericBuffer &&buf) noexcept
            : mSize{buf.mSize}, mCapacity{buf.mCapacity}, mType{buf.mType}, mAllocator{std::move(buf.mAllocator)}, mBuffer{buf.mBuffer}
        {
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }

        GenericBuffer &operator=(GenericBuffer &&buf) noexcept
        {
            if (this != &buf)
            {
                mAllocator.deallocate(mBuffer, toBytes(mCapacity));
                mSize = buf.mSize;
                mCapacity = buf.mCapacity;
                mType = buf.mType;
                mAllocator = std::move(buf.mAllocator);
                mBuffer = buf.mBuffer;
                buf.mSize = 0;
                buf.mCapacity = 0;
                buf.mBuffer = nullptr;
            }
            return *this;
        }

        [[nodiscard]] void *data() override
        {
            return mBuffer;
        }
        /*
        void generateRandomData() override
        {

            switch (mType)
            {
            case DataType::kFLOAT:
                mBuffer = static_cast<float *>(mBuffer);
                break;

            default:
                break;
            }

            for (std::size_t i = 0; i < mSize; ++i)
            {
                mBuffer[i] = static_cast<float>(rand()) / RAND_MAX;
            }
        }
        */

        [[nodiscard]] void const *data() const override
        {
            return mBuffer;
        }
        [[nodiscard]] std::size_t getSize() const override
        {
            return mSize;
        }
        [[nodiscard]] std::size_t getCapacity() const override
        {
            return mCapacity;
        }
        [[nodiscard]] DataType getDataType() const override
        {
            return mType;
        }
        [[nodiscard]] MemoryType getMemoryType() const override
        {
            return mAllocator.getMemoryType();
        }

        void resize(std::size_t newSize) override
        {
            if (mCapacity < newSize)
            {
                mAllocator.deallocate(mBuffer, toBytes(mCapacity));
                mBuffer = mAllocator.allocate(toBytes(newSize));
                mCapacity = newSize;
            }
            mSize = newSize;
        }

        void release() override
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
            mBuffer = nullptr;
            mSize = 0;
            mCapacity = 0;
        }

        ~GenericBuffer() override
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
        }

    protected:
        explicit GenericBuffer(std::size_t size, std::size_t capacity, DataType type, TAllocator allocator = {})
            : mSize{size}, mCapacity{capacity}, mType{type}, mAllocator{std::move(allocator)}, mBuffer{capacity > 0 ? mAllocator.allocate(toBytes(capacity)) : nullptr}
        {
        }

    private:
        std::size_t mSize{0}, mCapacity{0};
        DataType mType;
        TAllocator mAllocator;
        void *mBuffer;
    };

    using HostBuffer = GenericBuffer<HostAllocator>;

    template <typename TAllocator>
    class GenericTensor : public Tensor, public GenericBuffer<TAllocator>
    {
    public:
        using Base = GenericBuffer<TAllocator>;

        explicit GenericTensor(DataType type, TAllocator allocator = {})
            : Base{type, std::move(allocator)}
        {
            mDims.nbDims = 0;
        }

        explicit GenericTensor(Dims const &dims, DataType type, TAllocator allocator = {})
            : Base{volume(dims), type, std::move(allocator)}, mDims{dims}
        {
        }

        explicit GenericTensor(Dims const &dims, std::size_t capacity, DataType type, TAllocator allocator = {})
            : Base{volume(dims), capacity, type, std::move(allocator)}, mDims{dims}
        {
        }

        GenericTensor(GenericTensor &&tensor) noexcept
            : Base{std::move(tensor)}, mDims{tensor.mDims}
        {
            tensor.mDims.nbDims = 0;
        }

        GenericTensor &operator=(GenericTensor &&tensor) noexcept
        {
            if (this != &tensor)
            {
                Base::operator=(std::move(tensor));
                mDims = tensor.mDims;
                tensor.mDims.nbDims = 0;
            }
            return *this;
        }

        [[nodiscard]] Shape const &getShape() const override
        {
            return mDims;
        }

        void reshape(Shape const &dims) override
        {
            Base::resize(volume(dims));
            mDims = dims;
        }

        void release() override
        {
            Base::release();
            mDims.nbDims = 0;
        }

    private:
        Dims mDims{};
    };

    using HostTensor = GenericTensor<HostAllocator>;
}