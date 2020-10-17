#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>

#include "i_test_task_2d_pool.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class ITestTask2DThread
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ITestTask2DThread<T>>;
        
        // It can be called only once time, otherwise the behavior is undefined.
        virtual void Wait() = 0;

        virtual ~ITestTask2DThread() = default;

      };
    }
  }
}