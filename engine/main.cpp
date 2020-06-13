#include <iostream>
#include <vector>
#include <random>

#include "layer_2d.hpp"

int main(int argc, char** argv)
{
  try
  {
    cnn::ILayer2D<float>::Uptr layer_2d = std::make_unique<cnn::Layer2D<float>>(1, 10, 10, 10, 3, 3);
    layer_2d->Process();
  }
  catch (const std::exception& e)
  {
    std::cout << "Standard exception was caught: " << e.what() << std::endl;
  }
  catch (...)
  {
    std::cout << "Unknown exception was caught." << std::endl;
  }
  return 0;
}