#include <iostream>
using namespace std;

class A
{
public:
    A(int x) : m_x(x) {}
    int get_x() const { return m_x; }

private:
    int m_x;
};

class B
{
public:
    explicit B(int x) : m_x(x) {}
    int get_x() const { return m_x; }

private:
    int m_x;
};

void funcA(A a)
{
    cout << "a.get_x() = " << a.get_x() << endl;
}

void funcB(B b)
{
    cout << "b.get_x() = " << b.get_x() << endl;
}

int main()
{
    A a1 = 1;
    funcA(a1); // 可以隐式转换
    B b1(2);
    funcB(b1);                // 错误：只能显式转换
    funcB(static_cast<B>(3)); // 正确：显式转换
    return 0;
}