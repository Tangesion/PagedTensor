#pragma once
#include "common/dataType.h"
#include "block.h"

namespace toy::runtime
{
    class DataPtr
    {
    public:
        // offset is element number
        DataPtr(std::vector<Block *> *blockMap, size_t offsetStart)
            : blockMap(blockMap), offsetStart(offsetStart)
        {
            // std::cout << blockMap->at(0)->size << std::endl;
            size_t blockIndex = offsetStart / blockMap->at(0)->size;
            size_t blockOffset = offsetStart % blockMap->at(0)->size;
            data = static_cast<char *>(blockMap->at(blockIndex)->data) + blockOffset * blockMap->at(0)->typeSize;
        }
        DataPtr() : offsetStart(0), data(nullptr), blockMap(nullptr) {}

        ~DataPtr()
        {
            // std::cout << "DataPtr destroyed: offsetStart=" << offsetStart << ", data=" << data << std::endl;
            //  blockMap = nullptr;
            //  data = nullptr;
        }

        DataPtr operator+(size_t offset) const
        {
            size_t newOffsetStart = this->offsetStart + offset;
            size_t blockIndex = newOffsetStart / blockMap->at(0)->size;
            size_t blockOffset = newOffsetStart % blockMap->at(0)->size;
            void *newData = static_cast<char *>(blockMap->at(blockIndex)->data) + blockOffset * blockMap->at(0)->typeSize;
            return DataPtr(blockMap, newOffsetStart, newData);
        }

    private:
        // Private constructor to be used internally
        DataPtr(std::vector<Block *> *blockMap, size_t offsetStart, void *data)
            : blockMap(blockMap), offsetStart(offsetStart), data(data) {}

    public:
        size_t offsetStart;
        void *data;
        std::vector<Block *> *blockMap;
    };
};