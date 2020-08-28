#pragma once

#include <type_traits>
#include <memory>

#include "../convolution/i_map_2d.hpp"
#include "../common/i_map.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class ILesson2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ILesson2D<T>>;

        virtual size_t GetInputWidth() const = 0;
        virtual size_t GetIntputHeight() const = 0;
        virtual size_t GetInputCount() const = 0;

        virtual const convolution::IMap2D<T>& GetInput(const size_t index) const = 0;
        virtual convolution::IMap2D<T>& GetInput(const size_t index) = 0;

        virtual size_t GetOutputSize() const = 0;

        virtual const common::IMap<T>& GetOutput() const = 0;
        virtual common::IMap<T>& GetOutput() = 0;

        virtual ~ILesson2D() {}

      };
    }
  }
}