#include <cstddef>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>

int main()
{
    // std::string path = "/home/gexingt/tgx/models/Llama-2-7b-hf/pytorch_model-00001-of-00002.bin";
    std::string path = "/home/tgx/projects/paged_tensor/weight/test_weights_numpy.bin";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "无法打开文件: " << path << std::endl;
        return 1;
    }

    // 获取文件大小
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // 创建缓冲区
    std::vector<char> buffer(fileSize);

    // 读取文件内容到缓冲区
    if (!file.read(buffer.data(), fileSize))
    {
        std::cerr << "读取文件失败: " << path << std::endl;
        return 1;
    }

    // 关闭文件
    file.close();

    // 解析数值数据部分
    size_t numFloats = fileSize / sizeof(float);

    size_t numOffset = 32000 * 4 * 4096;
    const float *floatData = reinterpret_cast<const float *>(buffer.data() + numOffset);

    std::cout << "数值数据部分 (前 10 个 float):" << std::endl;
    for (size_t i = 0; i < 10 && i < numFloats; ++i)
    {
        std::cout << floatData[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
