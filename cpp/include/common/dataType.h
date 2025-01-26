#pragma once

#include <stdexcept>
#include <cstdint>

namespace toy::common
{

    enum class DataType : int32_t
    {
        kFLOAT = 0,
        kHALF = 1,
        kINT8 = 2,
        kINT32 = 3,
        kBOOL = 4,
        kUINT8 = 5,
        kFP8 = 6,
        kBF16 = 7,

        kINT64 = 8,
        kINT4 = 9,
    };

    constexpr static std::size_t getTypeSize(DataType type)
    {
        switch (type)
        {
        case DataType::kINT64:
            return 8;
        case DataType::kINT32:
            [[fallthrough]];
        case DataType::kFLOAT:
            return 4;
        case DataType::kBF16:
            [[fallthrough]];
        case DataType::kHALF:
            return 2;
        case DataType::kBOOL:
            [[fallthrough]];
        case DataType::kUINT8:
            [[fallthrough]];
        case DataType::kINT8:
            [[fallthrough]];
        case DataType::kFP8:
            return 1;
        case DataType::kINT4:
            std::runtime_error("Cannot determine size of INT4 data type");
        default:
            return 0;
        }
        return 0;
    }

} // namespace toy::common