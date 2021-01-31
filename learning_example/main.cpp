#include "Builder.hpp"

#include <iostream>

// TODO: We must add smart and fast autofit for topologies (It's very important for convenience!).

int main()
{
  try
  {
    auto library = cnn::learning_example::Builder<float>::GetLessonLibrary();
    auto network = cnn::learning_example::Builder<float>::GetNetwork();
    auto algorithm = cnn::learning_example::Builder<float>::GetGeneticAlgorithm();

    while (true)
    {
      auto newNetwork = algorithm.Run(library, network);
      std::swap(network, newNetwork);
    }
  }
  catch (const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }
  catch (...)
  {
    std::cout << "Unknown exception has been caught." << std::endl;
  }
  return 0;
}

