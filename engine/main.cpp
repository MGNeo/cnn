#include <iostream>
#include <time.h>

#include "network_2d.hpp"

int main(int argc, char** argv)
{
  try
  {
    const time_t t1 = clock();
    
    cnn::INetwork2D<float>::Uptr network_2d = std::make_unique<cnn::Network2D<float>>(3, 100, 100);

    network_2d->PushLayer(5, 5, 5);
    network_2d->PushLayer(10, 4, 4);
    network_2d->PushLayer(25, 3, 3);

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