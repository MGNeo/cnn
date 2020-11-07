#include <iostream>

#include "factory.hpp"

// Very important TODOs are:
// TODO: The serialization and deserialization must be full.
// TODO: All dynamic polymorphism must be killed because it leads to problems with serialization.
// TODO: Only standard activation function must be used.
// TODO: The complex network must be being created from a file or dynamic settings.
// TODO: Every convolution layer must have a pooler.
// TODO: All size must be multiple of two.

// Optionally TODOs are:
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