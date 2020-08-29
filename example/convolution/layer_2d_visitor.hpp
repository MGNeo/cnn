#pragma once

#include "../../engine/convolution/i_layer_2d_visitor.hpp"

#include "../../engine/convolution/convolution_layer_2d.hpp"
#include "../../engine/convolution/pooling_layer_2d.hpp"

#include <iostream>

namespace cnn
{
  namespace example
  {
    namespace convolution
    {
      template <typename T>
      class Layer2DVisitor : public engine::convolution::ILayer2DVisitor<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<Layer2DVisitor<T>>;

        void Visit(engine::convolution::PoolingLayer2D<T>& layer) override;
        void Visit(engine::convolution::ConvolutionLayer2D<T>& layer) override;

      };

      template <typename T>
      void Layer2DVisitor<T>::Visit(engine::convolution::PoolingLayer2D<T>& layer)
      {
        std::cout << "  " << __FUNCSIG__ << std::endl;

        std::cout << "    Layer type: " << typeid(layer).name() << std::endl;

        std::cout << "    Input width: " << layer.GetInputWidth() << std::endl;
        std::cout << "    Input height: " << layer.GetInputHeight() << std::endl;
        std::cout << "    Input count: " << layer.GetInputCount() << std::endl;

        std::cout << "    Output width: " << layer.GetOutputWidth() << std::endl;
        std::cout << "    Output height: " << layer.GetOutputHeight() << std::endl;
        std::cout << "    Output count: " << layer.GetOutputCount() << std::endl;

        // Here you can read and use all of the layer.
        // ...
      }

      template <typename T>
      void Layer2DVisitor<T>::Visit(engine::convolution::ConvolutionLayer2D<T>& layer)
      {
        std::cout << "  " << __FUNCSIG__ << std::endl;

        std::cout << "    Layer type: " << typeid(layer).name() << std::endl;

        std::cout << "    Input width: " << layer.GetInputWidth() << std::endl;
        std::cout << "    Input height: " << layer.GetInputHeight() << std::endl;
        std::cout << "    Input count: " << layer.GetInputCount() << std::endl;

        std::cout << "    Filter width: " << layer.GetFilterWidth() << std::endl;
        std::cout << "    Filter height: " << layer.GetFilterHeight() << std::endl;
        std::cout << "    Filter count: " << layer.GetFilterCount() << std::endl;

        std::cout << "    Output width: " << layer.GetOutputWidth() << std::endl;
        std::cout << "    Output height: " << layer.GetOutputHeight() << std::endl;
        std::cout << "    Output count: " << layer.GetOutputCount() << std::endl;

        // Here you can read and use all of the layer.
        // ...
      }
    }
  }
}