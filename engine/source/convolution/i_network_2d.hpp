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
      class INetwork2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<INetwork2D<T>>;

        //virtual void PushBack();
        virtual size_t GetLayerCount() const = 0;

        virtual ~INetwork2D() {}

      };
    }
  }
}