#include <iostream>

#include "convolution/convolution_handler_2d.hpp"

int main()
{
  cnn::engine::convolution::ConvolutionHandler2D<float> ch2d(1, 1, 1, 1, 1, 1);
  std::cout << "Hello World!\n";
}

