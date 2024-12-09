#include "runtime/BufferManager.h"
#include "runtime/Tensor.h"
#include "func/func.h"
#include <fstream>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>

using namespace inference_frame::runtime;

// int main()
//{
//     // Tensor::randTensor({16, 1, 4}, dataType);
//
//     Tensor::SharedPtr tensor = inference_frame::func::randTensor({2, 2, 3, 4}, DataType::kFLOAT, MemoryType::kCPU);
//     std::cout << *tensor << std::endl;
//     // tensor->reshape({4, 3, 4});
//     inference_frame::func::reShape(tensor, {5, 3, 4});
//     std::cout << *tensor << std::endl;
//
//     // std::ifstream f("../config/support.json");
//     // nlohmann::json data = nlohmann::json::parse(f);
//     // std::cout << data.dump(4) << std::endl;
//     // std::cout << data["dataType"][0] << std::endl;
//     //  auto dims = Tensor::makeShape({16, 1, 4});
//     //  auto constexpr dataType = DataType::kFLOAT;
//     //  Tensor::SharedPtr tensor{BufferManager::cpu(dims, dataType)};
//     //  tensor->printShape();
// }

TEST(TensorTest, test)
{
    Tensor::SharedPtr tensor = inference_frame::func::randTensor({2, 2, 3, 4}, DataType::kFLOAT, MemoryType::kCPU);
    std::cout << *tensor << std::endl;
    // tensor->reshape({4, 3, 4});
    inference_frame::func::reShape(tensor, {5, 3, 4});
    std::cout << *tensor << std::endl;
}