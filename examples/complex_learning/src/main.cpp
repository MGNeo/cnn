#include <iostream>

#include "factory.hpp"

// TODO: Add complex_using example .

// TODO: Add pooling processor into Layer2D.
// TODO: Add statistics (performance and other) in GeneticAlgorithm2D.
// TODO: Deny all move constructors and move operators=.
// TODO: Add marco ENABLE_STRONG_EXCEPTION_GUARANTEE.

// TODO: Add common statistic about the project (functions count, lines count, e.t.c.);

using namespace cnn::examples::complex_learning;

int main(int argc, char** argv)
{
  auto factory = std::make_unique<Factory<double>>();

  auto library = factory->Library();
  auto network = factory->Network();
  auto algorithm = factory->Algorithm();

  network = algorithm->Run(*library, *network);

  network->Save("weights.ws");
  network->Load("weights.ws");

  return 0;
}