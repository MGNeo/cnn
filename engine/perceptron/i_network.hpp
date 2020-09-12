#pragma once

#include <type_traits>
#include <memory>

#include "i_layer.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      template <typename T>
      class INetwork
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<INetwork<T>>;

        virtual size_t GetLayerCount() const = 0;

        virtual const ILayer<T>& GetLayer(const size_t index) const = 0;
        virtual ILayer<T>& GetLayer(const size_t index) = 0;

        virtual const ILayer<T>& GetLastLayer() const = 0;
        virtual ILayer<T>& GetLastLayer() = 0;

        virtual const ILayer<T>& GetFirstLayer() const = 0;
        virtual ILayer<T>& GetFirstLayer() = 0;

        virtual void Process() = 0;

        virtual void Accept(ILayerVisitor<T>& layerVisitor) = 0;

        // TODO: Clear() family methods.

        virtual typename INetwork<T>::Uptr Clone(const bool cloneState) const = 0;

        virtual ~INetwork() = default;

      };
    }
  }
}