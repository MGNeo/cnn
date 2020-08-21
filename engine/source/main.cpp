#include <iostream>

#include "convolution/network_2d.hpp"
#include "perceptron/network.hpp"
#include "complex/network_2d.hpp"

int main()
{
  // Create new 2D convolution network.
  // First layer of the network is convolution layer.
  cnn::engine::convolution::Network2D<float>::Uptr subNetwork2D = std::make_unique<cnn::engine::convolution::Network2D<float>>(32, 32, 3, 5, 5, 5);
  
  // Second layer of the network is pooling layer.
  subNetwork2D->PushBack(2);

  // Third layer of the network is convolution layer too.
  subNetwork2D->PushBack(8, 8, 15);

  // ---------------------------------------------------------------------------------

  // Create new perceptron network.
  // First layer of the network has 4 neurons.
  const size_t inputCount = subNetwork2D->GetOutputValueCount();
  cnn::engine::perceptron::Network<float>::Uptr subNetwork = std::make_unique<cnn::engine::perceptron::Network<float>>(inputCount, 4);

  // Second layer of the network has 8 neurons.
  subNetwork->PushBack(8);

  // Third layer of the network has 3 neurons.
  subNetwork->PushBack(3);

  // ---------------------------------------------------------------------------------

  // Create new complex network.
  cnn::engine::complex::INetwork2D<float>::Uptr network2D = std::make_unique<cnn::engine::complex::Network2D<float>>(std::move(subNetwork2D), std::move(subNetwork));

  // ---------------------------------------------------------------------------------

  // Example of using.
  {
    // Put some source into the first layer of the convolution network.
    {
      auto& firstLayer = network2D->GetConvolutionNetwork2D().GetFirstLayer();
      for (size_t i = 0; i < firstLayer.GetInputCount(); ++i)
      {
        auto& input = firstLayer.GetInput(i);
        for (size_t x = 0; x < firstLayer.GetInputWidth(); ++x)
        {
          for (size_t y = 0; y < firstLayer.GetInputHeight(); ++y)
          {
            // Some value for example.
            const float value = static_cast<float>(rand()) / RAND_MAX;
            input.SetValue(x, y, value);
          }
        }
      }
    }
    // Pass signal through the complex network.
    network2D->Process();
    // Take the result from the last layer of the perceptron network.
    const auto& outputLayer = network2D->GetPerceptronNetwork().GetLastLayer();
  }

  // TODO: Write complex::Lesson2D;
  // TODO: Write perceptron::ILayerVisitor (for the viewing of perceptron::ILayer with its real type "Layer" and like this);
  // TODO: Write convolution::ILayer2DVisitor (for the viewing of convolution::ILayer2D with its real type "PoolingLayer", "ConvolutionLayer" and like this);
  // TODO: Write genetic algorithm.
  // TODO: Add activation functions.
  // TODO: ...
  
  std::cout << "All was successfully completed!" << std::endl;
}

