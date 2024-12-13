#include <iostream>
#include <thread>

void foo(int &x)
{
    x += 1;
}

class A
{
public:
    void foo(int &x)
    {
        x += 1;
    }
    static void bar(int &x)
    {
        x += 1;
    }
};

int main()
{
    A a;
    int x = 5;
    std::thread t1(&A::foo, &a, std::ref(x));
    std::thread t2(A::bar, std::ref(x));
    // std::thread t1(foo, x);
    t1.join();
    t2.join();
    std::cout << "x = " << x << std::endl; // 输出修改后的x值
}