#pragma once

#include <memory>
#include <type_traits>

#include "i_pooling_handler_2d.hpp"
#include "i_convolution_handler_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class ILayer2D
      {
        
        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ILayer2D<T>>;

        virtual IPoolingHandler2D<T>& GetPoolingHandler() = 0;
        virtual const IPoolingHandler2D<T>& GetPoolingHandler() const = 0;

        virtual IConvolutionHandler2D<T>& GetConvolutionHandler() = 0;
        virtual const IConvolutionHandler2D<T>& GetConvolutionHandler() const = 0;

        virtual void Process() = 0;

        virtual ~ILayer2D() {}

      };
    }
  }
}