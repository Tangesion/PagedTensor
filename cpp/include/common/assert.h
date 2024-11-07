#include <stdexcept>
#include <iostream>

#define CHECK_WITH_INFO(cond, msg)         \
    do                                     \
    {                                      \
        if (!(cond))                       \
        {                                  \
            throw std::runtime_error(msg); \
        }                                  \
    } while (0)