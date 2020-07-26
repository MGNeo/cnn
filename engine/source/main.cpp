#include <iostream>
#include <time.h>

#include "perceptron.hpp"
#include "network_2d.hpp"
#include "lesson_2d.hpp"
#include "complex_network.hpp"

using namespace cnn;

int main(int argc, char** argv)
{
  // TODO: Think about exception safety in cnn::engine more carefully.
  try
  {
    const time_t t1 = clock();
    IActivator<float>::Uptr activator = std::make_unique<Activator<float>>();

    // Network2D.
    INetwork2D<float>::Uptr network2D = std::make_unique<Network2D<float>>(3, 32, 32);
    for (size_t i = 0; i < 15; ++i)
    {
      network2D->PushLayer(10, 3, 3, *activator);
    }

    // Perceptron.
    const auto& layer = network2D->GetLastLayer();
    const size_t count = layer.GetOutputCount() * layer.GetOutputWidth() * layer.GetOutputHeight();// Unsafe multiplication.

    IPerceptron<float>::Uptr perceptron = std::make_unique<Perceptron<float>>(count);
    perceptron->PushLayer(15);
    perceptron->PushLayer(25);
    perceptron->PushLayer(3);

    // Complex network.
    IComplexNetwork<float>::Uptr complexNetwork = std::make_unique<ComplexNetwork<float>>();
    complexNetwork->SetNetwork2D(std::move(network2D));
    complexNetwork->SetPerceptron(std::move(perceptron));
    complexNetwork->Process();

    // TODO: Create ComplexNetwork.
    // TODO: Create ComplexLesson.
    // TODO: Create genetic algorithm for ComplexNetwork.

    // TODO (Extended): Create lesson for Network2D.
    // TODO (Extended): Create genetic algorithm for Network2D.

    // TOD: (Extended): Create lesson for Perceptron.
    // TODO (Extended): Create genetic algorithm for Perceptron.

    const float dt = (clock() - t1) / (float)CLOCKS_PER_SEC;
    std::cout << dt << std::endl;
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