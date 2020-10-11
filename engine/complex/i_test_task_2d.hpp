#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>

#include "i_network_2d.hpp"
#include "i_lesson_2d_library.hpp"

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

        virtual void SetNetwork(INetwork2D<T>& network) = 0;
        virtual void SetLibrary(const ILesson2DLibrary<T>& library) = 0;

        virtual void Execute() = 0;

        virtual ~ITestTask2D() = default;
        
      };
    }
  }
}