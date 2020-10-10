#include <iostream>

#include "factory.hpp"

#include "engine/complex/network_2d.hpp"

using namespace cnn::examples::complex_learning;

int main(int argc, char** argv)
{
  IFactory<float>::Uptr factory = std::make_unique<Factory<float>>();

  auto library = factory->Library();
  auto network = factory->Network();
  auto algorithm = factory->Algorithm();

  algorithm->Run(*library, *network);

  return 0;
}