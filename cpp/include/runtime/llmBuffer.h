#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "block.h"
#include "common/dataPtr.h"
#include "buffer.h"
#include "common/dataType.h"
#include "tensor.h"
#include "common/assert.h"
using DataPtr = paged_tensor::common::DataPtr;
namespace paged_tensor::runtime
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

        void deallocate(void *ptr, std::size_t n)
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

    class HostPagedAllocator : public BaseAllocator<HostPagedAllocator, MemoryType::kCPU, false>
    {
        friend class BaseAllocator<HostPagedAllocator, MemoryType::kCPU, false>;

    public:
        HostPagedAllocator() noexcept = default;

    protected:
        // n is element number
        void allocateImpl(void **ptr, std::size_t n)
        {
            size_t blockSize = BlockManager::getInstance().getBlockSize();
            auto vec = new std::vector<void *>((n + blockSize - 1) / blockSize);
            for (size_t i = 0; i < vec->size(); i++)
            {
                vec->at(i) = BlockManager::getInstance().allocateBlock();
            }
            size_t offset = n % blockSize;
            *ptr = vec;
        }

        void deallocateImpl(void *ptr, std::size_t n)
        {
            auto vec = static_cast<std::vector<void *> *>(ptr);
            for (size_t i = 0; i < vec->size(); i++)
            {
                BlockManager::getInstance().free(vec->at(i));
            }
            delete vec;
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
        void allocateImpl(void **ptr, std::size_t n) // NOLINT(readability-convert-member-functions-to-static)
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

    // TODO: Now just kv cache wrapper
    template <MemoryType memoryType>
    class BorrowingPagedKVAllocator : public BaseAllocator<BorrowingPagedKVAllocator<memoryType>, memoryType, false>
    {
        friend class BaseAllocator<BorrowingPagedKVAllocator<memoryType>, memoryType, false>;

    public:
        using Base = BaseAllocator<BorrowingPagedKVAllocator<memoryType>, memoryType, false>;

        BorrowingPagedKVAllocator(size_t layerIdx, size_t length, bool isKey, bool isWrapNewBlock)
            : mLayerIdx(layerIdx), mLength(length), mIsKey(isKey), mIsWrapNewBlock(isWrapNewBlock)
        {
            CHECK_WITH_INFO(mLength == 1 || !mIsWrapNewBlock, "when wrapNewBlock is True, length must to be 1");
        }

    protected:
        void allocateImpl(void **ptr, std::size_t n) // NOLINT(readability-convert-member-functions-to-static)
        {
            *ptr = KVCacheManager::getInstance().helpWrapper(mLayerIdx, mLength, mIsKey, mIsWrapNewBlock);
        }

        void deallocateImpl( // NOLINT(readability-convert-member-functions-to-static)
            [[maybe_unused]] void *ptr, [[maybe_unused]] std::size_t n)
        {
        }

    private:
        size_t mLayerIdx;
        size_t mLength;
        bool mIsKey;
        bool mIsWrapNewBlock;
    };

    using CpuBorrowingAllocator = BorrowingAllocator<MemoryType::kCPU>;
    using CpuPagedKVBorrowingAllocator = BorrowingPagedKVAllocator<MemoryType::kCPU>;

    template <typename TAllocator>
    class GenericBuffer : virtual public Buffer
    {
    public:
        using AllocatorType = TAllocator;

        // Construct an empty buffer
        explicit GenericBuffer(
            DataType type, TAllocator allocator = {})
            : GenericBuffer{0, type, std::move(allocator)} {
              };

        // Construct a buffer with a given size

        explicit GenericBuffer(
            std::size_t size, DataType type, TAllocator allocator = {}, bool paged = false)
            : GenericBuffer{size, size, type, std::move(allocator), paged} {
              };

        GenericBuffer(GenericBuffer &&buf) noexcept
            : mSize{buf.mSize}, mCapacity{buf.mCapacity}, mType{buf.mType}, mAllocator{std::move(buf.mAllocator)}, mBuffer{buf.mBuffer}, mDataPtr{buf.mDataPtr}, mPaged{buf.mPaged}
        {
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
            buf.mDataPtr = DataPtr();
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
                mPaged = buf.mPaged;
                mDataPtr = buf.mDataPtr;
                buf.mSize = 0;
                buf.mCapacity = 0;
                buf.mDataPtr = DataPtr();
                buf.mBuffer = nullptr;
            }
            return *this;
        }
        [[nodiscard]] DataPtr dataPaged() const override
        {
            return mDataPtr;
        }

        [[nodiscard]] void *data() override
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
                mDataPtr = DataPtr(getBlockMap(), 0, BlockManager::getInstance().getBlockSize(), BlockManager::getInstance().getTypeSize());
            }
            mSize = newSize;
        }

        void release() override
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
            mBuffer = nullptr;
            mSize = 0;
            mCapacity = 0;
            if (mPaged)
            {
                mDataPtr = DataPtr();
            }
        }

        [[nodiscard]] bool isPaged() const override
        {
            return mPaged;
        }

        [[nodiscard]] std::vector<void *> *getBlockMap() override
        {
            return static_cast<std::vector<void *> *>(mBuffer);
        }

        ~GenericBuffer() override
        {
            mAllocator.deallocate(mBuffer, toBytes(mCapacity));
        }

    protected:
        explicit GenericBuffer(std::size_t size, std::size_t capacity, DataType type, TAllocator allocator = {}, bool paged = false)
            : mSize{size}, mCapacity{capacity}, mType{type}, mAllocator{std::move(allocator)}, mBuffer{capacity > 0 ? (!paged ? mAllocator.allocate(toBytes(capacity)) : mAllocator.allocate(capacity)) : nullptr}, mPaged{paged}
        {
            if (mPaged)
            {
                mDataPtr = DataPtr(getBlockMap(), 0, BlockManager::getInstance().getBlockSize(), BlockManager::getInstance().getTypeSize());
            }
        }

    private:
        std::size_t mSize{0}, mCapacity{0};
        DataType mType;
        TAllocator mAllocator;
        DataPtr mDataPtr;
        bool mPaged{false};
        void *mBuffer;
    };

    using HostBuffer = GenericBuffer<HostAllocator>;
    using HostPagedBuffer = GenericBuffer<HostPagedAllocator>;

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

        explicit GenericTensor(Dims const &dims, DataType type, bool paged, TAllocator allocator = {})
            : Base{volume(dims), type, std::move(allocator), paged}, mDims{dims}
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
            // std::cout << "Derived!" << std::endl;
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
    using HostPagedTensor = GenericTensor<HostPagedAllocator>;
}