#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>



namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class ITestTask2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:
        
        using Uptr = std::unique_ptr<ITestTask2D<T>>;

        virtual void Execute() = 0;

        virtual ~ITestTask2D() = default;
        
      };
    }
  }
}