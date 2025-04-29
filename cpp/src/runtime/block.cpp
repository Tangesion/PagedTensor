#include "runtime/block.h"
#include <iostream>

namespace paged_tensor::runtime
{
    // BlockManager 实现

    BlockManager &BlockManager::getInstance()
    {
        static BlockManager instance;
        return instance;
    }

    void BlockManager::initialize(size_t blockNum, size_t blockSize, DataType type)
    {
        if (memoryPool == nullptr)
        {
            // allocate memory pool with size
            memoryPool = std::make_unique<char[]>(blockNum * blockSize * paged_tensor::common::getTypeSize(type));
            freeBlocks.reserve(blockNum);
            char *rawPtr = memoryPool.get();
            for (size_t i = 0; i < blockNum; i++)
            {
                freeBlocks.push_back(static_cast<void *>(rawPtr + i * blockSize * paged_tensor::common::getTypeSize(type)));
            }
            this->blockSize = blockSize;
            this->blockNum = blockNum;
            this->type = type;
            this->typeSize = paged_tensor::common::getTypeSize(type);
        }
    }

    void BlockManager::destroyInstance()
    {
        BlockManager &instance = getInstance();
        instance.freeBlocks.clear();
        instance.freeBlocks.shrink_to_fit();
        instance.memoryPool.reset(); // Release the memory pool
        instance.additionalPools.clear();
        instance.additionalPools.shrink_to_fit();
    }

    void BlockManager::extend()
    {
        // std::cout << "pre extend block num " << blockNum << std::endl;
        // std::cout << "extend!" << std::endl;
        size_t newBlockNum = blockNum;
        std::unique_ptr<char[]> newMemoryPool = std::make_unique<char[]>(newBlockNum * blockSize * typeSize);
        char *rawPtr = newMemoryPool.get();

        for (size_t i = 0; i < newBlockNum; i++)
        {
            freeBlocks.push_back(static_cast<void *>(rawPtr + i * blockSize * typeSize));
        }

        additionalPools.push_back(std::move(newMemoryPool));

        blockNum += newBlockNum;
        // std::cout << "after extend block num " << blockNum << std::endl;
    }

    void *BlockManager::allocateBlock()
    {
        if (freeBlocks.empty())
        {
            extend();
        }

        void *block = freeBlocks.back();
        freeBlocks.pop_back();
        return block;
    }

    void BlockManager::free(void *block)
    {
        freeBlocks.push_back(block);
    }

    size_t BlockManager::getBlockSize()
    {
        return blockSize;
    }

    size_t BlockManager::getTypeSize()
    {
        return typeSize;
    }

    DataType BlockManager::getType()
    {
        return type;
    }

    void BlockManager::printStatus() const
    {
        size_t totalBlocks = blockNum;
        size_t freeBlockCount = freeBlocks.size();
        size_t usedBlockCount = totalBlocks - freeBlockCount;
        size_t mainPoolSize = blockSize * typeSize * (blockNum - (additionalPools.empty() ? 0 : additionalPools.size() * blockNum / (additionalPools.size() + 1)));
        size_t additionalPoolsSize = 0;

        for (size_t i = 0; i < additionalPools.size(); i++)
        {
            additionalPoolsSize += blockSize * typeSize * blockNum / (i + 2);
        }

        size_t totalMemorySize = mainPoolSize + additionalPoolsSize;

        std::cout << "BlockManager Status:" << std::endl;
        std::cout << "  Total Blocks: " << totalBlocks << std::endl;
        std::cout << "  Used Blocks: " << usedBlockCount << std::endl;
        std::cout << "  Free Blocks: " << freeBlockCount << std::endl;
        std::cout << "  Block Size: " << blockSize << " elements (" << blockSize * typeSize << " bytes)" << std::endl;
        std::cout << "  Data Type: " << " (" << typeSize << " bytes)" << std::endl;
        std::cout << "  Main Memory Pool: " << mainPoolSize << " bytes" << std::endl;
        std::cout << "  Additional Memory Pools: " << additionalPoolsSize << " bytes" << std::endl;
        std::cout << "  Total Memory: " << totalMemorySize << " bytes (" << totalMemorySize / (1024.0 * 1024.0) << " MB)" << std::endl;
        std::cout << std::endl;
    }

    BlockManager::~BlockManager()
    {
        destroyInstance();
    }

    // KVCacheManager 实现

    KVCacheManager &KVCacheManager::getInstance()
    {
        static KVCacheManager instance;
        return instance;
    }

    void KVCacheManager::initialize(size_t layerNums)
    {
        this->layerNums = layerNums;
        kvLayerLists.resize(layerNums);
        cacheLength.resize(layerNums);
        for (size_t i = 0; i < layerNums; i++)
        {
            kvLayerLists[i] = new HeadTail();
            kvLayerLists[i]->head = new KVListNode();
            cacheLength[i] = 0;
        }
    }

    void KVCacheManager::allocate(size_t layerIdx, size_t length)
    {
        KVListNode *curr = kvLayerLists[layerIdx]->tail ? kvLayerLists[layerIdx]->tail : kvLayerLists[layerIdx]->head;
        for (size_t i = 0; i < length; i++)
        {
            void *kBlock = BlockManager::getInstance().allocateBlock();
            void *vBlock = BlockManager::getInstance().allocateBlock();
            curr->next = new KVListNode(kBlock, vBlock);
            curr = curr->next;
        }
        kvLayerLists[layerIdx]->tail = curr;
        cacheLength[layerIdx] += length;
    }

    void KVCacheManager::destroy()
    {
        for (size_t i = 0; i < layerNums; i++)
        {
            KVListNode *curr = kvLayerLists[i]->head->next;
            while (curr)
            {
                BlockManager::getInstance().free(curr->kBlock);
                BlockManager::getInstance().free(curr->vBlock);
                curr = curr->next;
            }
        }
    }

    size_t KVCacheManager::getCacheLength(size_t layerIdx)
    {

        if (layerIdx >= cacheLength.size())
        {
            std::cerr << "Error: Invalid layer index " << layerIdx
                      << ", max is " << (cacheLength.size() - 1) << std::endl;
            return 0; // 返回安全值而不是崩溃
        }
        return cacheLength[layerIdx];
    }

    std::vector<void *> *KVCacheManager::helpWrapper(size_t layerid, size_t length, bool isKey, bool isWrapNewBlock)
    {

        auto vec = new std::vector<void *>(length);
        if (!isWrapNewBlock)
        {
            KVListNode *curr = kvLayerLists[layerid]->head->next;
            for (size_t i = 0; i < length; i++)
            {
                vec->at(i) = isKey ? curr->kBlock : curr->vBlock;
                curr = curr->next;
            }
        }
        else
        {
            vec->at(0) = isKey ? kvLayerLists[layerid]->tail->kBlock : kvLayerLists[layerid]->tail->vBlock;
        }
        return vec;
    }

    void KVCacheManager::printStatus() const
    {
        std::cout << "KVCacheManager Status:" << std::endl;
        std::cout << "  Number of Layers: " << layerNums << std::endl;

        size_t totalKVPairs = 0;
        size_t totalMemoryUsed = 0;
        size_t blockSize = BlockManager::getInstance().getBlockSize();
        size_t typeSize = BlockManager::getInstance().getTypeSize();
        size_t bytesPerBlock = blockSize * typeSize;

        std::cout << "  Per-Layer KV Cache:" << std::endl;
        for (size_t i = 0; i < layerNums; i++)
        {
            size_t kvLength = 0;
            KVListNode *curr = kvLayerLists[i]->head->next;

            while (curr)
            {
                kvLength++;
                curr = curr->next;
            }

            totalKVPairs += kvLength;
            size_t layerMemory = kvLength * 2 * bytesPerBlock; // *2 because each KV pair has K and V blocks
            totalMemoryUsed += layerMemory;

            std::cout << "    Layer " << i << ": " << kvLength << " KV pairs ("
                      << (layerMemory / 1024.0) << " KB)" << std::endl;
        }

        std::cout << "  Total KV Pairs: " << totalKVPairs << std::endl;
        std::cout << "  Block Size: " << blockSize << " elements (" << bytesPerBlock << " bytes)" << std::endl;
        std::cout << "  Total Memory Used: " << (totalMemoryUsed / (1024.0 * 1024.0)) << " MB" << std::endl;

        // Calculate memory efficiency
        if (totalKVPairs > 0)
        {
            size_t avgBytesPerToken = totalMemoryUsed / totalKVPairs;
            std::cout << "  Avg Memory per KV Pair: " << avgBytesPerToken / (1024.0) << " KB" << std::endl;
        }

        std::cout << std::endl;
    }
}
