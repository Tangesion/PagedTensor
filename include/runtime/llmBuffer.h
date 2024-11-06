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

        void allocate(std::size_t n)
        {
            void *ptr{};
            static_cast<TDerived *>(this)->allocateImpl(&ptr, n);
            if constexpr (count)
            {
                MemoryCounters::getInstance().allocate<memoryType>(n);
            }
            return ptr;
        }

        void deallocate(void *ptr, std::size_t n)
        {
            if (ptr)
            {
                static_cast<TDerived *>(this)->deallocateImpl(ptr, n);
                if constexpr (count)
                {
                    MemoryCounters::getInstance().deallocate<memoryType>(n);
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
            [[maybe_unused]] PointerType ptr, [[maybe_unused]] std::size_t n)
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
    };

    template <typename TAllocator>
    class GenericBuffer : virtual public Buffer
    {
    public:
        using AllocatorType = TAllocator;

        // Construct an empty buffer
        explicit GenericBuffer(
            std::size_t size, DataType type, TAllocator allocator = {})
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

        void *data() override
        {
            return mBuffer;
        }

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
}