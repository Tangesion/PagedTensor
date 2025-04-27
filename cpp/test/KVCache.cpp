#include <gtest/gtest.h>
#include <chrono>
#include <func/func.h>

using namespace paged_tensor::func;
using namespace paged_tensor::runtime;

#define BLOCK_SIZE 4096
#define BLOCK_NUM 4096
#define Layer_NUM 32

TEST(KVCacheTest, allocateTest)
{
    BlockManager::getInstance().initialize(BLOCK_NUM, BLOCK_SIZE, DataType::kFLOAT);
    KVCacheManager::getInstance().initialize(Layer_NUM);
    BlockManager::getInstance().printStatus();
    // prefill allocate
    size_t length = 4096;
    size_t layerIdx = 15;
    KVCacheManager::getInstance().allocate(layerIdx, length);
    BlockManager::getInstance().printStatus();
    KVCacheManager::getInstance().printStatus();

    // decode allocate
    KVCacheManager::getInstance().allocate(layerIdx, 1);
    KVCacheManager::getInstance().printStatus();
}

TEST(KVCacheTest, wrapTest)
{
    BlockManager::getInstance().initialize(BLOCK_NUM, BLOCK_SIZE, DataType::kFLOAT);
    KVCacheManager::getInstance().initialize(Layer_NUM);
    BlockManager::getInstance().printStatus();
    // prefill allocate
    size_t length = 4096;
    size_t layerIdx = 15;
    size_t bsz = 1;
    size_t headNums = 32;
    size_t headDims = 128;

    KVCacheManager::getInstance().allocate(layerIdx, length);
    BlockManager::getInstance().printStatus();
    KVCacheManager::getInstance().printStatus();

    Tensor::UniquePtr wrapVPrefill;
    Tensor::Shape shape = Tensor::makeShape({Tensor::DimType64(bsz), Tensor::DimType64(length), Tensor::DimType64(headNums), Tensor::DimType64(headDims)});
    wrapVPrefill = Tensor::kvCacheWrap(DataType::kFLOAT, shape, length, layerIdx, false, false);
    wrapVPrefill->printShape();
    std::cout << "ispaged :" << wrapVPrefill->isPaged() << std::endl;

    KVCacheManager::getInstance().allocate(layerIdx, 1);
    KVCacheManager::getInstance().printStatus();

    // decode allocate
    Tensor::UniquePtr wrapKNew;
    shape = Tensor::makeShape({Tensor::DimType64(bsz), Tensor::DimType64(1), Tensor::DimType64(headNums), Tensor::DimType64(headDims)});
    wrapKNew = Tensor::kvCacheWrap(DataType::kFLOAT, shape, 1, layerIdx, true, true);
    wrapKNew->printShape();
    std::cout << "ispaged :" << wrapKNew->isPaged() << std::endl;

    Tensor::UniquePtr wrapKNewAll;
    shape = Tensor::makeShape({Tensor::DimType64(bsz), Tensor::DimType64(length + 1), Tensor::DimType64(headNums), Tensor::DimType64(headDims)});
    wrapKNewAll = Tensor::kvCacheWrap(DataType::kFLOAT, shape, length + 1, layerIdx, true, false);
    wrapKNewAll->printShape();
}