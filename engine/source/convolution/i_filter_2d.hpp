#pragma once

#include <memory>
#include <type_traits>

#include "i_core_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class IFilter2D
      {
        
        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<IFilter2D<T>>;

        virtual size_t GetWidth() const = 0;
        virtual size_t GetHeight() const = 0;

        virtual size_t GetCoreCount() const = 0;

        virtual ICore2D<T>& GetCore(const size_t index) = 0;
        virtual const ICore2D<T>& GetCore(const size_t index) const = 0;

        virtual ~IFilter2D() {}

      };
    }
  }
}