#pragma once
#include <cstdint>
#include <memory>
#include <queue>
#include <vector>
#include <iostream>
#include "common/dataType.h"
#include "buffer.h"
#include "common/assert.h"

namespace paged_tensor::runtime
{

    class BlockManager
    {
    public:
        static BlockManager &getInstance();

        void initialize(size_t blockNum, size_t blockSize, DataType type);
        static void destroyInstance();
        void extend();
        void *allocateBlock();
        void free(void *block);
        size_t getBlockSize();
        size_t getTypeSize();
        DataType getType();
        void printStatus() const;

    private:
        BlockManager() = default;
        ~BlockManager();
        BlockManager(BlockManager const &) = delete;
        BlockManager &operator=(BlockManager const &) = delete;

    private:
        std::vector<void *> freeBlocks;
        std::unique_ptr<char[]> memoryPool;
        std::vector<std::unique_ptr<char[]>> additionalPools;
        size_t blockSize;
        size_t blockNum;
        DataType type;
        size_t typeSize;
    };

    struct KVListNode
    {
        KVListNode *next;
        void *kBlock;
        void *vBlock;
        KVListNode(void *kBlock, void *vBlock) : kBlock(kBlock), vBlock(vBlock), next(nullptr) {}
        KVListNode() : kBlock(nullptr), vBlock(nullptr), next(nullptr) {}
    };

    struct HeadTail
    {
        KVListNode *head;
        KVListNode *tail;
        HeadTail() : head(nullptr), tail(nullptr) {}
    };

    class KVCacheManager
    {
    public:
        static KVCacheManager &getInstance();

        void initialize(size_t layerNums);
        void allocate(size_t layerIdx, size_t length);
        void destroy();
        size_t getCacheLength(size_t layerIdx);
        std::vector<void *> *helpWrapper(size_t layerid, size_t length, bool isKey, bool isWrapNewBlock);
        void printStatus() const;

    private:
        KVCacheManager() = default;
        ~KVCacheManager() = default;
        KVCacheManager &operator=(KVCacheManager const &) = delete;
        KVCacheManager(KVCacheManager const &) = delete;

    private:
        std::vector<HeadTail *> kvLayerLists;
        std::vector<size_t> cacheLength;
        size_t layerNums;
    };
};