#include <iostream>

#include "factory.hpp"

#include "engine/complex/network_2d.hpp"

// TODO: Algorithm must be changed because the existing doesn't work.

// TODO: Add pooling processor into Layer2D.
// TODO: Add statistics (performance and other) in GeneticAlgorithm2D.
// TODO: Deny all move constructors and move operators=.
// TODO: Add marco ENABLE_STRONG_EXCEPTION_GUARANTEE.

using namespace cnn::examples::complex_learning;

int main(int argc, char** argv)
{
  auto factory = std::make_unique<Factory<double>>();

  auto library = factory->Library();
  auto network = factory->Network();
  auto algorithm = factory->Algorithm();

  while (true)
  {
    network = algorithm->Run(*library, *network);
  }

  return 0;
}