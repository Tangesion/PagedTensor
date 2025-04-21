#pragma once
#include <cstdint>
#include <memory>
#include <queue>
#include "common/dataType.h"
#include "buffer.h"
#include "common/assert.h"
namespace paged_tensor::runtime
{

    // class Block
    //{
    // public:
    //     using DataType = paged_tensor::common::DataType;
    //     Block(size_t size, DataType type, void *data)
    //         : size{size}, typeSize{paged_tensor::common::getTypeSize(type)}, type{type}, data{data}, offset{0}
    //     {
    //     }
    //     ~Block() = default;
    //
    // public:
    //    size_t size; // num of elements
    //    size_t typeSize;
    //    DataType type;
    //    size_t offset; // offset num in block
    //    void *data;
    //};
    class BlockManager
    {
    public:
        static BlockManager &getInstance()
        {
            static BlockManager instance;
            return instance;
        }

        void initialize(size_t blockNum, size_t blockSize, DataType type)
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
        static void destroyInstance()
        {
            BlockManager &instance = getInstance();
            instance.freeBlocks.clear();
            instance.freeBlocks.shrink_to_fit();
            instance.memoryPool.reset(); // Release the memory pool
            instance.additionalPools.clear();
            instance.additionalPools.shrink_to_fit();
        }

        void extend()
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

        void *allocateBlock()
        {
            // TODO: implement expansion in the future
            // try
            //{
            //    CHECK_WITH_INFO(!freeBlocks.empty(), "No free block available");
            //}
            // catch (const std::exception &e)
            //{
            //    std::cerr << e.what() << '\n';
            //    std::exit(EXIT_FAILURE);
            //}
            if (freeBlocks.empty())
            {
                extend();
            }

            void *block = freeBlocks.back();
            freeBlocks.pop_back();
            return block;
        }

        void free(void *block)
        {
            freeBlocks.push_back(block);
        }

    private:
        BlockManager() = default;
        ~BlockManager() = default;
        BlockManager(BlockManager const &) = delete;
        BlockManager &operator=(BlockManager const &) = delete;

    public:
        std::vector<void *> freeBlocks;
        std::unique_ptr<char[]> memoryPool;
        std::vector<std::unique_ptr<char[]>> additionalPools;
        size_t blockSize;
        size_t blockNum;
        DataType type;
        size_t typeSize;
    };
};