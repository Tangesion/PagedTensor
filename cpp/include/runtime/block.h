#pragma once
#include <cstdint>
#include <memory>
#include <queue>
#include "buffer.h"
#include "common/dataType.h"
namespace toy::runtime
{

    class Block
    {
    public:
        using DataType = toy::common::DataType;
        Block(size_t size, DataType type, void *memoryPool)
            : size{size}, typeSize{toy::common::getTypeSize(type)}, type{type}, data{data}
        {
        }
        ~Block() = default;

    public:
        size_t size; // num of elements
        size_t typeSize;
        DataType type;
        void *data;
    };

    class BlockManager
    {
    public:
        using SharedPtr = std::shared_ptr<Block>;
        using UniquePtr = std::unique_ptr<Block>;
        static BlockManager &getInstance(size_t size)
        {
            static BlockManager instance;
            if (instance.memoryPool == nullptr)
            {
                instance.memoryPool = std::make_unique<void>(size);
            }
            return instance;
        }

    private:
        BlockManager() = default;
        ~BlockManager() = default;
        BlockManager(BlockManager const &) = delete;
        BlockManager &operator=(BlockManager const &) = delete;

    public:
        std::queue<UniquePtr> freeBlocks;
        std::unique_ptr<void> memoryPool;
    };
};