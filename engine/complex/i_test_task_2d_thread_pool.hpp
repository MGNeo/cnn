#pragma once

#include "i_test_task_2d_thread.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class ITestTask2DThreadPool
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ITestTask2DThreadPool<T>>;

        // It can be called from any thread any time.
        virtual bool IsWrong() const = 0;

        virtual ~ITestTask2DThreadPool() = default;

      };
    }
  }
}