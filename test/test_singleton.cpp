
#include <iostream>
class Singleton
{
public:
    // 获取唯一实例的静态方法
    static Singleton &getInstance()
    {
        static Singleton instance; // 唯一实例，使用局部静态变量实现延迟初始化
        return instance;
    }

    // 公共方法
    void doSomething()
    {
        // 实现具体功能
    }

private:
    // 私有化构造函数，防止外部创建实例
    Singleton() = default;

    // 删除拷贝构造函数和赋值运算符，防止拷贝和赋值
    Singleton(const Singleton &) = delete;
    Singleton &operator=(const Singleton &) = delete;
};

int main()
{
    // 获取单例实例并调用方法
    Singleton &instance = Singleton::getInstance();
    instance.doSomething();

    return 0;
}