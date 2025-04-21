#include <iostream>
#include <chrono>
#include <vector>
#include <memory>

// 简单的内存池实现
class SimpleMemoryPool
{
private:
    char *memoryBlock;
    size_t blockSize;
    size_t chunkSize;
    std::vector<void *> freeChunks;

public:
    SimpleMemoryPool(size_t totalSize, size_t chunkSize)
        : blockSize(totalSize), chunkSize(chunkSize)
    {
        // 分配大块内存
        memoryBlock = new char[totalSize];

        // 将内存分割成多个chunk并管理起来
        freeChunks.reserve(totalSize / chunkSize);
        for (size_t i = 0; i < totalSize; i += chunkSize)
        {
            freeChunks.push_back(memoryBlock + i);
        }
    }

    ~SimpleMemoryPool()
    {
        delete[] memoryBlock;
    }

    void *allocate()
    {
        if (freeChunks.empty())
        {
            std::cout << "no free chunk!" << std::endl;
            return nullptr; // 内存池已满
        }

        void *chunk = freeChunks.back();
        freeChunks.pop_back();
        return chunk;
    }

    void deallocate(void *ptr)
    {
        // 简单检查指针是否在内存池范围内
        if (ptr >= memoryBlock && ptr < memoryBlock + blockSize)
        {
            freeChunks.push_back(ptr);
        }
    }
};

int main()
{
    constexpr size_t NUM_ALLOCS = 100;  // 分配次数
    constexpr size_t CHUNK_SIZE = 4096; // 每次分配大小
    constexpr size_t NORM_SIZE = 4096 * 4;
    constexpr size_t POOL_SIZE = NORM_SIZE * NUM_ALLOCS; // 内存池大小
    constexpr size_t BLOCK_NUM = NORM_SIZE * NUM_ALLOCS / CHUNK_SIZE;

    // 模拟场景1: 多次连续分配和释放 (不利于系统分配器)
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < NUM_ALLOCS; ++i)
        {
            void *ptr = malloc(NORM_SIZE);
            free(ptr);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Regular malloc/free (many small allocations): "
                  << duration.count() << " seconds" << std::endl;
    }

    // 模拟场景2: 使用内存池连续分配和释放
    {
        SimpleMemoryPool pool(POOL_SIZE, CHUNK_SIZE);

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < BLOCK_NUM; ++i)
        {
            void *ptr = pool.allocate();
            pool.deallocate(ptr);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Memory pool allocate/deallocate: "
                  << duration.count() << " seconds" << std::endl;
    }

    // 模拟场景3: 批量分配然后批量释放 (更真实的场景)
    {

        std::vector<void *> ptrs;
        ptrs.reserve(NUM_ALLOCS);
        auto start = std::chrono::high_resolution_clock::now();
        // 批量分配
        for (size_t i = 0; i < NUM_ALLOCS; ++i)
        {
            ptrs.push_back(malloc(NORM_SIZE));
        }

        // 批量释放
        for (void *ptr : ptrs)
        {
            free(ptr);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Regular malloc/free (batch): "
                  << duration.count() << " seconds" << std::endl;
    }

    // 模拟场景4: 使用内存池批量分配和释放
    {
        SimpleMemoryPool pool(POOL_SIZE, CHUNK_SIZE);

        auto start = std::chrono::high_resolution_clock::now();

        std::vector<void *> ptrs;
        ptrs.reserve(BLOCK_NUM);

        // 批量分配
        for (size_t i = 0; i < BLOCK_NUM; ++i)
        {
            ptrs.push_back(pool.allocate());
        }

        // 批量释放
        for (void *ptr : ptrs)
        {
            pool.deallocate(ptr);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Memory pool (batch): "
                  << duration.count() << " seconds" << std::endl;
    }

    return 0;
}