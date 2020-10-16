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

        // The best interface is a interface which has nothing!

        virtual ~ITestTask2DThread() = default;

      };
    }
  }
}