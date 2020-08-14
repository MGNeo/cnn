#include <iostream>

#include "convolution/pooling_layer_2d.hpp"
#include "convolution/convolution_layer_2d.hpp"

using namespace cnn::engine::convolution;

int main()
{
  ILayer2D<float>::Uptr layer{};
  layer = std::make_unique<PoolingLayer2D<float>>(10, 10, 3, 3);
  layer = std::make_unique<ConvolutionLayer2D<float>>(10, 10, 10, 3, 3, 3);

  std::cout << "Hello World!\n";
}

