#include <vector>
#include <iostream>

class MyClass
{
public:
    MyClass() : x_(0)
    {
        std::cout << "Default constructing MyClass\n";
    }
    MyClass(int x) : x_(x)
    {
        std::cout << "Constructing MyClass(" << x_ << ")\n";
    }
    MyClass(const MyClass &other) : x_(other.x_)
    {
        std::cout << "Copy constructing MyClass(" << x_ << ")\n";
    }
    MyClass(MyClass &&other) noexcept : x_(other.x_)
    {
        std::cout << "Move constructing MyClass(" << x_ << ")\n";
    }

private:
    int x_;
};

int main()
{
    std::vector<MyClass> vec(10);
    MyClass a(1);
    vec.push_back(MyClass(2));    // 移动构造
    vec.emplace_back(MyClass(3)); // 移动构造
    vec.push_back(a);             // 拷贝构造
    vec.emplace_back(a);          // 拷贝构造

    vec.push_back(std::move(a));    // 移动构造
    vec.emplace_back(std::move(a)); // 移动构造

    vec.push_back(4);    // 移动构造
    vec.emplace_back(5); // 末尾直接插入，无需移动构造

    return 0;
}