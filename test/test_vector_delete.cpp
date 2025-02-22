#include <iostream>
#include <vector>

class A
{
public:
    A() { std::cout << "A()" << std::endl; }
    ~A() { std::cout << "~A()" << std::endl; }
};

int main()
{
    std::vector<A *> *vec = new std::vector<A *>(2);
    vec->at(0) = new A();
    vec->at(1) = new A();
    delete vec;
    return 0;
}