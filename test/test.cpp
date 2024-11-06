#include <memory>
#include <vector>
#include <iostream>

#include <iostream>

// CRTP base class template
template <typename Derived>
class Animal
{
public:
    void speak() const
    {
        // Calls the function in the derived class
        static_cast<const Derived *>(this)->speakImpl();
    }
};

// Derived class implementing the speakImpl function
class Dog : public Animal<Dog>
{
public:
    void speakImpl() const
    {
        std::cout << "Dog barks" << std::endl;
    }
};

// Another derived class
class Cat : public Animal<Cat>
{
public:
    void speakImpl() const
    {
        std::cout << "Cat meows" << std::endl;
    }
};
template <typename T>
void interface(Animal<T> &animal)
{
    animal.speak();
}

int test_crtp()
{
    Dog dog;
    Cat cat;

    dog.speak(); // Outputs: Dog barks
    cat.speak(); // Outputs: Cat meows

    // Attempting runtime polymorphism
    Animal<Dog> *animalPtr = &dog;
    // Animal<Cat> *animalPtr = &cat;
    animalPtr->speak();
    // Cannot create an array of base class pointers
    // Animal* animals[] = { &dog, &cat }; // Not possible
    interface(dog);

    return 0;
}

template <typename Derived>
class Base
{
public:
    void interface()
    {
        auto *ptr = static_cast<Derived *>(this)->implementation();
    }
};

class Derived : public Base<Derived>
{
public:
    int *implementation()
    {
        return nullptr;
    }
};

class IBuffer
{
public:
    [[nodiscard]] virtual void *data() = 0;

    //!
    //! \brief Returns a pointer to underlying array.
    //!
    [[nodiscard]] virtual void const *data() const = 0;

    //!
    //! \brief Returns a pointer to the underlying array at a given element index.
    //!
    [[nodiscard]] virtual void *data(std::size_t index)
    {
        auto *const dataPtr = this->data();
        // std::cout << "dataPtr before cast: " << dataPtr << std::endl;
        //  cout this address
        std::cout << "IBuffer func: " << this << std::endl;

        if (dataPtr)
        {
            auto *castedPtr = static_cast<std::uint8_t *>(dataPtr) + 1;
            // std::cout << "dataPtr after cast: " << static_cast<void *>(castedPtr) << std::endl;
            // std::cout << "base" << std::endl;
            return castedPtr;
        }
        return nullptr;
    }
    virtual int test()
    {
        std::cout << "test" << std::endl;
    }

    //!
    //! \brief Returns a pointer to the underlying array at a given element index.
    //!
    [[nodiscard]] virtual void const *data(std::size_t index) const
    {
        auto const *const dataPtr = this->data();

        return dataPtr ? static_cast<std::uint8_t const *>(dataPtr) + 1 : nullptr;
    }

    virtual ~IBuffer() = default;

    //!
    //! \brief Not allowed to copy.
    //!
    IBuffer(IBuffer const &) = delete;

    //!
    //! \brief Not allowed to copy.
    //!
    IBuffer &operator=(IBuffer const &) = delete;

    IBuffer() = default;
};

class BufferView : virtual public IBuffer
{
public:
    explicit BufferView(std::shared_ptr<IBuffer> buffer, std::size_t offset, std::size_t size)
        : mBuffer(std::move(buffer)), mOffset{offset}, mSize{size}
    {
    }

    // void *data() override
    //{
    //     std::cout << "BufferView" << std::endl;
    //
    //    return mBuffer->data(mOffset);
    //}
    //
    //[[nodiscard]] void const *data() const override
    //{
    //    return mBuffer->data(mOffset);
    //}

private:
    std::shared_ptr<IBuffer> mBuffer;
    std::size_t mOffset, mSize;
};

class MockBuffer : public IBuffer
{
public:
    MockBuffer(std::vector<uint8_t> data) : mData(std::move(data)) {}

    void *data() override
    {
        std::cout << "MockBuffer func:" << this << std::endl;
        std::cout << "MockBuffer" << std::endl;
        return mData.data();
    }

    [[nodiscard]] void const *data() const override
    {
        return mData.data();
    }
    void test_derived()
    {
    }

private:
    std::vector<uint8_t> mData;
};

enum class DataType
{
    kFloat32,
    kInt32,
    // 其他数据类型
};

class ITensor
{
public:
    virtual ~ITensor() = default;

    virtual DataType getDataType() const = 0; // 纯虚函数
};

class DerivedTensor : public ITensor
{
public:
    DataType getDataType() const override
    {
        return DataType::kFloat32;
    }
};

void func(IBuffer *buffer)
{
    buffer->data();
}

// template <typename C>
// class B
//{
//     typedef typename C::T T; // 期望 C 有一个类型 T
//     T *p_;
// };
//
// class D : public B<D>
//{
// public:
//     typedef int T; // 将 T 定义为类型
// };

void testBufferView()
{
    std::vector<uint8_t> bufferData = {1, 2, 3, 4, 5};
    auto mockBuffer = std::make_shared<MockBuffer>(bufferData);
    // mockBuffer->data(2);
    std::size_t offset = 1;
    mockBuffer->test();
    IBuffer *buffer = mockBuffer.get();
    buffer->data(offset);
    // void *value = mockBuffer->data(offset);
    //  BufferView bufferView(mockBuffer, 1, 3);
    //   bufferView.data();
    //  func(&bufferView);
    //   bufferView.test();
    //
    //    const uint8_t *data = static_cast<const uint8_t *>(bufferView.data());
    //    if (data[0] == 2 && data[1] == 3 && data[2] == 4)
    //{
    //       std::cout << "Test passed!" << std::endl;
    //   }
    //    else
    //{
    //       std::cout << "Test failed!" << std::endl;
    //   }
}

void test_std_move()
{

    std::string str1 = "aacasxs";
    std::vector<std::string> vec;
    std::cout << str1 << std::endl;
    std::string str2 = std::move(str1); // 调用拷贝构造函数
    std::cout << std::move(str1) << std::endl;
    std::cout << std::move(str2) << std::endl;

    int a = 5;
    std::cout << a << std::endl;
    int b = std::move(a);
    std::cout << a << std::endl;
    // 输出
    // aacasxs
    // null
    // aacasxs
    // 5
    // 5
}

int main()
{
    // testBufferView();
    // test_std_move();
    test_crtp();
    return 0;
}