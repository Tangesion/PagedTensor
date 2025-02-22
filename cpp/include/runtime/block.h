#pragma once
#include <cstdint>
#include <memory>
#include <queue>
#include "common/dataType.h"
#include "buffer.h"
#include "common/assert.h"
namespace toy::runtime
{

    class Block
    {
    public:
        using DataType = toy::common::DataType;
        Block(size_t size, DataType type, void *data)
            : size{size}, typeSize{toy::common::getTypeSize(type)}, type{type}, data{data}, offset{0}
        {
        }
        ~Block() = default;

    public:
        size_t size; // num of elements
        size_t typeSize;
        DataType type;
        size_t offset; // offset num in block
        void *data;
    };

    class BlockManager
    {
    public:
        using SharedPtr = std::shared_ptr<Block>;
        using UniquePtr = std::unique_ptr<Block>;
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
                memoryPool = std::make_unique<char[]>(blockNum * blockSize * toy::common::getTypeSize(type));
                for (size_t i = 0; i < blockNum; i++)
                {
                    char *rawPtr = memoryPool.get();
                    freeBlocks.push(std::make_unique<Block>(blockSize, type, static_cast<void *>(rawPtr + i * blockSize * toy::common::getTypeSize(type))));
                }
                this->blockSize = blockSize;
                this->blockNum = blockNum;
                // std::cout << freeBlocks.size() << std::endl;
            }
        }
        static void destroyInstance()
        {
            BlockManager &instance = getInstance();
            instance.freeBlocks = std::queue<UniquePtr>(); // Clear the queue
            instance.memoryPool.reset();                   // Release the memory pool
        }

        UniquePtr allocateBlock()
        {
            // TODO: implement wait in the future
            try
            {
                CHECK_WITH_INFO(!freeBlocks.empty(), "No free block available");
            }
            catch (const std::exception &e)
            {
                std::cerr << e.what() << '\n';
                std::exit(EXIT_FAILURE);
            }
            UniquePtr block = std::move(freeBlocks.front());
            freeBlocks.pop();
            return block;
        }

        void free(UniquePtr block)
        {
            freeBlocks.push(std::move(block));
        }

    private:
        BlockManager() = default;
        ~BlockManager() = default;
        BlockManager(BlockManager const &) = delete;
        BlockManager &operator=(BlockManager const &) = delete;

    public:
        std::queue<UniquePtr> freeBlocks;
        std::unique_ptr<char[]> memoryPool;
        size_t blockSize;
        size_t blockNum;
    };
};