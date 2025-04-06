// #pragma once
// #include "common/dataType.h"
// #include "block.h"
//
// namespace paged_tensor::runtime
//{
//     class DataPtr
//     {
//     public:
//         // offset is element number
//         DataPtr(std::vector<void *> *blockMap, size_t offsetStart)
//             : blockMap(blockMap), offsetStart(offsetStart)
//         {
//         }
//         DataPtr() : offsetStart(0), blockMap(nullptr) {}
//
//         ~DataPtr()
//         {
//         }
//
//         DataPtr operator+(size_t offset) const
//         {
//             size_t newOffsetStart = this->offsetStart + offset;
//             return DataPtr(blockMap, newOffsetStart);
//         }
//
//         DataPtr operator[](size_t offset) const
//         {
//             size_t newOffsetStart = this->offsetStart + offset;
//             return DataPtr(blockMap, newOffsetStart);
//         }
//
//         DataPtr &operator++()
//         {
//             this->offsetStart++;
//             return *this;
//         }
//
//         DataPtr operator++(int)
//         {
//             DataPtr temp = *this;
//             ++(*this);
//             return temp;
//         }
//
//     public:
//         void *data()
//         {
//             size_t blockSize = BlockManager::getInstance().blockSize;
//             size_t typeSize = BlockManager::getInstance().typeSize;
//             size_t blockIndex = offsetStart / blockSize;
//             size_t blockOffset = offsetStart % blockSize;
//             return static_cast<void *>(static_cast<char *>(blockMap->at(blockIndex)) + blockOffset * typeSize);
//         }
//
//     private:
//         size_t offsetStart;
//         std::vector<void *> *blockMap;
//     };
// };