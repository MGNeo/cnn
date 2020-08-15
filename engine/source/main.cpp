#include <iostream>

#include "convolution/pooling_layer_2d.hpp"
#include "convolution/convolution_layer_2d.hpp"
#include "convolution/network_2d.hpp"

#include "perceptron/layer.hpp"
#include "perceptron/network.hpp"

int main()
{
  {
    using namespace cnn::engine::convolution;

    ILayer2D<float>::Uptr layer2d = std::make_unique<ConvolutionLayer2D<float>>(32, 32, 1, 4, 4, 16);

    // TODO: Think about more clear constructor.
    INetwork2D<float>::Uptr network2d = std::make_unique<Network2D<float>>(std::move(layer2d));

    network2d->PushBack(2);
    network2d->PushBack(7, 7, 64);
    network2d->PushBack(3);
    network2d->PushBack(3, 3, 256);

    network2d->Process();
  }

  {
    using namespace cnn::engine::perceptron;
    ILayer<float>::Uptr layer = std::make_unique<Layer<float>>(10, 5);

    // TODO: Think about more clear constructor.
    INetwork<float>::Uptr network = std::make_unique<Network<float>>(std::move(layer));

    network->PushBack(8);
    network->PushBack(11);
    network->PushBack(3);

    network->Process();
  }

  std::cout << "Hello World!\n";
}

