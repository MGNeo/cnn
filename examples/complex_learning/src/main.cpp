#include <iostream>

#include "loader.hpp"

using namespace cnn::examples::complex_learning;

int main(int argc, char** argv)
{
  ILoader<float>::Uptr loader = std::make_unique<Loader<float>>();

  auto library = loader->Load();

  return 0;
}