#include "common/Neuron.hpp"

#include "common/Map.hpp"

#include "convolution/Map2D.hpp"

#include "convolution/Core2D.hpp"

#include "convolution/Filter2DTopology.hpp"
#include "convolution/Filter2D.hpp"

#include "convolution/Layer2DTopology.hpp"
#include "convolution/Layer2D.hpp"

#include "convolution/Network2DTopology.hpp"
#include "convolution/Network2D.hpp"

#include "perceptron/LayerTopology.hpp"
#include "perceptron/Layer.hpp"

#include "perceptron/NetworkTopology.hpp"
#include "perceptron/Network.hpp"

#include "common/Mutagen.hpp"

#include "complex/Network2DTopology.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

// TODO: Perhaps, Neuron must use Map instead of [] for Inputs and Weights.
// TODO: Check that any move-assignment operators reset "from".

using namespace cnn::engine;

int main(int argc, char** argv)
{
  common::Neuron<float> neuron;

  common::Map<float> map;

  convolution::Map2D<float> map2D;

  convolution::Core2D<float> core;

  convolution::Filter2DTopology filterTopology{ {10, 10}, 10 };
  convolution::Filter2D<float> filter{ filterTopology };

  convolution::Layer2DTopology layer2DTopology{ {10, 10}, 10, { {3, 3}, 10 }, 5, {8, 8}, 5 };
  convolution::Layer2D<float> layer2D{ layer2DTopology };

  convolution::Network2DTopology network2DTopology;
  network2DTopology.PushBack(layer2DTopology);
  network2DTopology.PushBack(layer2DTopology);
  network2DTopology.Reset();

  convolution::Network2D<float> network2D{ network2DTopology };
  network2D.GenerateOputput();

  perceptron::LayerTopology layerTopology{ 10, 10 };
  perceptron::Layer<float> layer{ layerTopology };

  perceptron::NetworkTopology networkTopology;
  networkTopology.PushBack(layerTopology);
  networkTopology.PushBack(layerTopology);

  perceptron::Network<float> network { networkTopology };

  complex::Network2DTopology complexNetwork2DTopology{ network2DTopology, networkTopology };

  std::fstream file("C:/Users/MGNeo/Desktop/network2DTopology.raw", std::ios_base::out | std::ios_base::binary);
  complexNetwork2DTopology.Save(file);

  // TODO: Check save/load correctness.

  common::Mutagen<float> mutagen;
  convolution::Size2D size;

  return 0;
}