#pragma once

#include <type_traits>
#include <memory>

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      // Pre-declarations...
      template <typename T>
      class Layer;

      template <typename T>
      class ILayerVisitor
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ILayerVisitor<T>>;

        virtual void Visit(Layer<T>& layer) = 0;

        // TODO: Perhaps, it needs to create "const" methods too.

        virtual ~ILayerVisitor() = default;

      };
    }
  }
}