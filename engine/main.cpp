#include <iostream>
#include <time.h>

#include "network_2d.hpp"

int main(int argc, char** argv)
{
  // TODO: Think about exception safety in cnn::engine more carefully.
  try
  {
    const time_t t1 = clock();
    cnn::IActivator<float>::Uptr activator = std::make_unique<cnn::Activator<float>>();

    cnn::INetwork2D<float>::Uptr network_2d = std::make_unique<cnn::Network2D<float>>(3, 32, 32);

    for (size_t i = 0; i < 15; ++i)
    {
      network_2d->PushLayer(15, 3, 3, *activator);
    }
    
    // TODO: Add fully connected layer (ILayerFC, LayerFC).

    network_2d->Process();

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