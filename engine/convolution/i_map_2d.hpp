#pragma once

#include <memory>
#include <type_traits>

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class IMap2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<IMap2D<T>>;

        virtual size_t GetWidth() const = 0;
        virtual size_t GetHeight() const = 0;

        virtual T GetValue(const size_t x, const size_t y) const = 0;
        virtual void SetValue(const size_t x, const size_t y, const T value) = 0;

        virtual void Clear() = 0;

        virtual ~IMap2D() = default;

      };
    }
  }
}