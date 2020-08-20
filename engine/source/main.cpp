#include <iostream>

#include "convolution/extensible_network_2d.hpp"
#include "perceptron/extensible_network.hpp"
#include "complex/network_2d.hpp"

int main()
{
  // Create new extensible 2D convolution network.
  // First layer of the network is convolution layer.
  cnn::engine::convolution::ExtensibleNetwork2D<float>::Uptr eNetwork2D = std::make_unique<cnn::engine::convolution::ExtensibleNetwork2D<float>>(32, 32, 3, 5, 5, 25);
  
  // Second layer of the network is pooling layer.
  eNetwork2D->PushBack(2);

  // Third layer of the network is convolution layer too.
  eNetwork2D->PushBack(8, 8, 150);

  // ---------------------------------------------------------------------------------

  // Create new extensible perceptron network.
  // First layer of the network has 4 neurons.
  const size_t inputCount = eNetwork2D->GetOutputValueCount();
  cnn::engine::perceptron::ExtensibleNetwork<float>::Uptr eNetwork = std::make_unique<cnn::engine::perceptron::ExtensibleNetwork<float>>(inputCount, 4);

  // Second layer of the network has 8 neurons.
  eNetwork->PushBack(8);

  // Third layer of the network has 3 neurons.
  eNetwork->PushBack(3);

  // ---------------------------------------------------------------------------------

  // Create new complex network.
  cnn::engine::complex::INetwork2D<float>::Uptr network2D = std::make_unique<cnn::engine::complex::Network2D<float>>(std::move(eNetwork2D), std::move(eNetwork));

  // TODO: Write example of using.
  // ...
  network2D->Process();
  
  std::cout << network2D->GetConvolutionNetwork2D().GetOutputValueCount() << std::endl;

  std::cout << "Hello World!\n";
}

