#include <iostream>

#include "convolution/pooling_layer_2d.hpp"
#include "convolution/convolution_layer_2d.hpp"
#include "convolution/network_2d.hpp"

using namespace cnn::engine::convolution;

int main()
{
  ILayer2D<float>::Uptr layer = std::make_unique<ConvolutionLayer2D<float>>(32, 32, 1, 4, 4, 16);

  INetwork2D<float>::Uptr network = std::make_unique<Network2D<float>>(std::move(layer));

  network->PushBack(2);
  network->PushBack(7, 7, 64);
  network->PushBack(3);
  network->PushBack(3, 3, 256);

  network->Process();

  std::cout << "Hello World!\n";
}

