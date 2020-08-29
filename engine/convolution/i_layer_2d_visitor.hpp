#pragma once

#include <type_traits>
#include <memory>

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // Pre-declarations...

      template <typename T>
      class PoolingLayer2D;

      template <typename T>
      class ConvolutionLayer2D;

      template <typename T>
      class ILayer2DVisitor
      {
        static_assert(std::is_floating_point<T>::value);

      public:
        
        using Uptr = std::unique_ptr<ILayer2DVisitor<T>>;

        virtual void Visit(PoolingLayer2D<T>& layer) = 0;
        virtual void Visit(ConvolutionLayer2D<T>& layer) = 0;

        // TODO: Perhaps, it needs to create "const" methods too.

        virtual ~ILayer2DVisitor() {}

      };
    }
  }
}