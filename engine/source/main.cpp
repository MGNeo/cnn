#include <iostream>

#include "convolution/layer_2d.hpp"

int main()
{
  cnn::engine::convolution::Layer2D<float> layer2d{ 10, 10, 3, 3, 3, 3, 20 };
  layer2d.Process();
  std::cout << "Hello World!\n";
}

