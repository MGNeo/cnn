#pragma once

#include "../../engine/perceptron/i_layer_visitor.hpp"
#include <iostream>

namespace cnn
{
  namespace example
  {
    namespace perceptron
    {
      template <typename T>
      class LayerVisitor : public engine::perceptron::ILayerVisitor<T>
      {
        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<LayerVisitor<T>>;

        void Visit(engine::perceptron::Layer<T>& layer) override;

      };

      template <typename T>
      void LayerVisitor<T>::Visit(engine::perceptron::Layer<T>& layer)
      {
        std::cout << "Input size: " << layer.GetInputSize() << std::endl;
        std::cout << "Output size: " << layer.GetOutputSize() << std::endl;
        // Here you can read and use all from the layer.
        // ...
      }
    }
  }
}