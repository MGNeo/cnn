#include <iostream>

#include "core.hpp"
#include "core2d.hpp"
#include "layer.hpp"
#include "layer2d.hpp"

int main(int argc, char** argv)
{
  try
  {
    const cnn::engine::Core<float, 10> core;
    core.GetInput(0);

    cnn::engine::Core2D<double, 5, 20> core2d;
    core2d.SetWeight(3, 3, 3);

    cnn::engine::Layer<float, 10> layer;
    layer.SetValue(1, 1);

    cnn::engine::Layer2D<long double, 10, 10> layer2d;
    layer2d.SetValue(3, 3, 100);

  }
  catch (const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
  catch (...)
  {
    std::cout << "Unknown exception was caught." << std::endl;
  }
  return 0;
}