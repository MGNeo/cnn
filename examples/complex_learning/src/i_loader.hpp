#pragma once

#include <memory>
#include <cstdint>
#include <cstddef>
#include <string>

#include "engine/complex/i_lesson_2d_library.hpp"

namespace cnn
{
  namespace examples
  {
    namespace complex_learning
    {
      template <typename T>
      class ILoader
      {

        static_assert(std::is_floating_point<T>::value);
        
      public:

        using Uptr = std::unique_ptr<ILoader<T>>;

        virtual typename engine::complex::ILesson2DLibrary<T>::Uptr Load() const = 0;

        virtual ~ILoader() = default;

      };
    }
  }
}