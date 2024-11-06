#include <iostream>

class MyClass
{
public:
    MyClass(int *p, std::string name) : ptr(p), name(name)
    {
        std::cout << "Constructor: " << name << std::endl;
    }
    ~MyClass()
    {
        std::cout << "Destructor" << name << std::endl;
    }
    MyClass(MyClass &other) : ptr(other.ptr), name(std::move(other.name))
    {
        other.ptr = nullptr;
        std::cout << "Move Constructor" << std::endl;
    }

    int *ptr;
    std::string name;
};

int main()
{
    int *p = new int(10);
    MyClass a(p, "a");
    MyClass b(a);
    std::cout << "a.ptr: " << a.ptr << std::endl;
    std::cout << "b.ptr: " << b.ptr << std::endl;
    return 0;
}