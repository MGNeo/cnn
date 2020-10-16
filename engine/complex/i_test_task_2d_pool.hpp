#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>

#include "i_test_task_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class ITestTask2DPool
      {

        static_assert(std::is_floating_point<T>::value);

      public:
        
        using Uptr = std::unique_ptr<ITestTask2DPool<T>>;

        // If the pool is empty the method returns nullptr.
        // Exception guarantee: strong
        virtual typename ITestTask2D<T>::Uptr Pop() = 0;

        // Return true, if the pool is empty.
        virtual bool IsEmpty() const = 0;

        virtual ~ITestTask2DPool() = default;

      };
    }
  }
}