#pragma once
#include "block.h"

namespace toy::runtime
{
    class DataPtr
    {
    public:
        // offset is element number
        DataPtr(std::shared_ptr<std::vector<Block *>> blockMap, size_t offsetStart)
            : blockMap(blockMap), offsetStart(offsetStart)
        {
            std::cout << blockMap->at(0)->size << std::endl;
            size_t blockIndex = offsetStart / blockMap->at(0)->size;
            size_t blockOffset = offsetStart % blockMap->at(0)->size;
            data = static_cast<char *>(blockMap->at(blockIndex)->data) + blockOffset * blockMap->at(0)->typeSize;
        }
        DataPtr() : offsetStart(0), data(nullptr), blockMap(nullptr) {}

        ~DataPtr()
        {
            blockMap = nullptr;
            data = nullptr;
        }

        DataPtr operator+(size_t offset)
        {
            DataPtr ptr;
            ptr.offsetStart += offsetStart + offset;
            size_t blockIndex = ptr.offsetStart / blockMap->at(0)->size;
            size_t blockOffset = ptr.offsetStart % blockMap->at(0)->size;
            ptr.data = static_cast<char *>(blockMap->at(blockIndex)->data) + blockOffset * blockMap->at(0)->typeSize;
            ptr.blockMap = blockMap;
            return ptr;
        }

    public:
        size_t offsetStart;
        void *data;
        std::shared_ptr<std::vector<Block *>> blockMap;
    };
};