#pragma once

#include "../convolution/Map2D.hpp"
#include "../common/Map.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      /*
      template <typename T>
      class Lesson2D
      {

        static_assert(std::is_floating_point<T>::value)

      public:

        Lesson2D(const convolution::Map2DTopology& inputTopology = {},
                 const common::MapTopology& outputTOpology = {});

        Lesson2D(const Lesson2D& lesson);

        Lesson2D(Lesson2D&& lesson) noexcept;

        Lesson2D& operator=(const Lesson2D& lesson);

        Lesson2D& operator=(Lesson2D&& lesson) noexcept;

      private:

        convolution::Map2D<T> Input;
        common::Map<T> Output;

      };
      */
    }
  }
}