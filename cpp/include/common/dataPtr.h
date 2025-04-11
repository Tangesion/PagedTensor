#pragma once
#include <vector>
#include <cstdint>
#include <stdexcept>

namespace paged_tensor::common
{

    class DataPtr
    {
    public:
        // offset is element number
        DataPtr(std::vector<void *> *blockMap, size_t offsetStart, size_t blockSize, size_t typeSize)
            : blockMap(blockMap), offsetStart(offsetStart), blockSize(blockSize), typeSize(typeSize)
        {
        }
        DataPtr() : offsetStart(0), blockMap(nullptr) {}

        ~DataPtr()
        {
        }

        DataPtr operator+(size_t offset) const
        {
            size_t newOffsetStart = this->offsetStart + offset;
            return DataPtr(blockMap, newOffsetStart, this->blockSize, this->typeSize);
        }

        DataPtr operator[](size_t offset) const
        {
            size_t newOffsetStart = this->offsetStart + offset;
            return DataPtr(blockMap, newOffsetStart, this->blockSize, this->typeSize);
        }

        DataPtr &operator++()
        {
            this->offsetStart++;
            return *this;
        }

        DataPtr operator++(int)
        {
            DataPtr temp = *this;
            ++(*this);
            return temp;
        }

    public:
        size_t getBlockOffset()
        {
            return offsetStart % blockSize;
        }

        size_t getBlockSize()
        {
            return blockSize;
        }

        size_t getBlockIdx()
        {
            return offsetStart / blockSize;
        }

    public:
        template <typename T>
        T *data()
        {
            size_t blockIndex = offsetStart / blockSize;
            size_t blockOffset = offsetStart % blockSize;
            return static_cast<T *>(static_cast<void *>(static_cast<char *>(blockMap->operator[](blockIndex)) + blockOffset * typeSize));
        }
        // T *data()
        //{
        //     static void *cachedBlock = nullptr;
        //     static size_t cachedBlockIndex = -1;
        //
        //    size_t blockIndex = offsetStart / blockSize;
        //    if (blockIndex != cachedBlockIndex)
        //    {
        //        cachedBlock = blockMap->at(blockIndex);
        //        cachedBlockIndex = blockIndex;
        //    }
        //
        //    size_t blockOffset = offsetStart % blockSize;
        //    return static_cast<T *>(static_cast<void *>(static_cast<char *>(cachedBlock) + blockOffset * typeSize));
        //}

    private:
        size_t offsetStart;
        size_t blockSize;
        size_t typeSize;
        std::vector<void *> *blockMap;
    };
};