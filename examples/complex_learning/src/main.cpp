#include <iostream>

#include "factory.hpp"

#include "engine/complex/network_2d.hpp"

// TODO: Add multithreading to GeneticAlgorithm2D.
  // GeneticAlgorithm2D<T>::Test().
// TODO: Add statistics (performance and other) in GeneticAlgorithm2D.
// TODO: Deny all move constructors and move operators=.
// TODO: Add marco ENABLE_STRONG_EXCEPTION_GUARANTEE.

using namespace cnn::examples::complex_learning;

int main(int argc, char** argv)
{
  IFactory<float>::Uptr factory = std::make_unique<Factory<float>>();

  //auto library = factory->Library();
  //auto network = factory->Network();
  //auto algorithm = factory->Algorithm();

  //auto newNetwork = algorithm->Run(*library, *network);

  return 0;
}