#include <iostream>

template <typename T>
void wrapper(T &&u)
{
    if (std::is_lvalue_reference<decltype(u)>::value)
    {
        std::cout << "u is an lvalue\n";
    }
    else if (std::is_rvalue_reference<decltype(u)>::value)
    {
        std::cout << "u is an rvalue\n";
    }
    else
    {
        std::cout << "u is neither an lvalue nor an rvalue\n";
    }
    fun(u);
}

class MyClass
{
public:
    int data; // 增加成员变量以增加对象的复杂性

    MyClass() : data(0) { std::cout << "in MyClass()\n"; }
    MyClass(const MyClass &other) : data(other.data) { std::cout << "in MyClass(const MyClass&)\n"; }
    MyClass(MyClass &&other) noexcept : data(other.data) { std::cout << "in MyClass(MyClass&&)\n"; }
    MyClass &operator=(const MyClass &other)
    {
        data = other.data;
        std::cout << "in MyClass operator=(const MyClass&)\n";
        return *this;
    }
    MyClass &operator=(MyClass &&other) noexcept
    {
        data = other.data;
        std::cout << "in MyClass operator=(MyClass&&)\n";
        return *this;
    }
};

void fun(MyClass &a) { std::cout << "in fun(MyClass&)\n"; }
void fun(const MyClass &a) { std::cout << "in fun(const MyClass&)\n"; }
void fun(MyClass &&a)
{
    std::cout << "in fun(MyClass &&)\n";
    if (std::is_lvalue_reference<decltype(a)>::value)
    {
        std::cout << "a is an lvalue\n";
    }
    else if (std::is_rvalue_reference<decltype(a)>::value)
    {
        std::cout << "a is an rvalue\n";
    }
    else
    {
        std::cout << "a is neither an lvalue nor an rvalue\n";
    }
}

int main(void)
{
    MyClass a;
    const MyClass b;

    fun(a);
    fun(b);
    fun(MyClass());

    std::cout << "----- Wrapper ------\n";
    wrapper(a);
    wrapper(b);
    wrapper(MyClass());

    MyClass &&rr1 = MyClass();

    const MyClass &rr2 = MyClass();

    return 0;
}
