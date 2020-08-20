#pragma once

#include <memory>
#include <type_traits>

#include "i_layer_2d.hpp"

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
        
        virtual size_t GetLayerCount() const = 0;

        virtual const ILayer2D<T>& GetLayer(const size_t index) const = 0;
        virtual ILayer2D<T>& GetLayer(const size_t index) = 0;

        virtual const ILayer2D<T>& GetLastLayer() const = 0;
        virtual ILayer2D<T>& GetLastLayer() = 0;

        virtual size_t GetOutputValueCount() const = 0;

        virtual void Process() = 0;

        virtual ~INetwork2D() {}

      };
    }
  }
}