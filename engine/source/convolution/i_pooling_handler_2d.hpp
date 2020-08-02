#pragma once

#include <memory>
#include <type_traits>

#include "map_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class IPoolingHandler2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<IPoolingHandler2D<T>>;

        virtual size_t GetInputWidth() const = 0;
        virtual size_t GetInputHeight() const = 0;

        virtual size_t GetStepSize() const = 0;

        virtual size_t GetOutputWidth() const = 0;
        virtual size_t GetOutputHeight() const = 0;

        virtual size_t GetChannelCount() const = 0;

        virtual IMap2D<T>& GetInput(const size_t index) = 0;
        virtual const IMap2D<T>& GetInput(const size_t index) const = 0;

        virtual IMap2D<T>& GetOutput(const size_t index) = 0;
        virtual const IMap2D<T>& GetOutput(const size_t index) const = 0;

        virtual void Process() = 0;

        virtual void Clear() = 0;

        virtual ~IPoolingHandler2D() {}

      };
    }
  }
}