#include <iostream>
#include <vector>
#include <random>

#include "core_2d.hpp"
#include "matrix_2d.hpp"

int main(int argc, char** argv)
{
  try
  {
    /*
    cnn::IMatrix2D<float>::Uptr matrix = std::make_unique<cnn::Matrix2D<float>>(10, 10);
    matrix->SetCell(4, 5, 5.f);

    std::default_random_engine e;
    std::uniform_int_distribution<size_t> d{ 1, 10 };

    std::vector<cnn::ICore2D<float>::Uptr> cores_2d;

    for (size_t i = 0; i < 10; ++i)
    {
      cores_2d.emplace_back(std::make_unique<cnn::Core2D<float>>(d(e), d(e)));
    }

    for (auto& core_2d : cores_2d)
    {
      std::cout << core_2d->GetWidth() << std::endl;
    }

    // TODO: Layer.
    // TODO: Network.
    */
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