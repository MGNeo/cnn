#include "common/Neuron.hpp"
#include "common/RefNeuron.hpp"
#include "common/ConstRefNeuron.hpp"

#include "common/Map.hpp"
#include "common/RefMap.hpp"
#include "common/ConstRefMap.hpp"

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
#include "perceptron/RefLayer.hpp"
#include "perceptron/ConstRefLayer.hpp"

#include "perceptron/NetworkTopology.hpp"
#include "perceptron/Network.hpp"

#include "common/Mutagen.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

// TODO: Perhaps, Neuron must use Map instead of [] for Inputs and Weights.
// TODO: Check that any move-assignment operators reset "from".

using namespace cnn::engine;

int main(int argc, char** argv)
{
  common::Neuron<float> neuron;
  common::RefNeuron<float> refNeuron{ neuron };
  common::ConstRefNeuron<float> constRefNeuron{ neuron };

  common::Map<float> map;
  common::RefMap<float> refMap{ map };
  common::ConstRefMap<float> constRefMap{ map };

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
  perceptron::RefLayer<float> refLayer{ layer };
  perceptron::ConstRefLayer<float> constRefLayer{ layer };

  perceptron::NetworkTopology networkTopology;
  networkTopology.PushBack(layerTopology);
  networkTopology.PushBack(layerTopology);

  perceptron::Network<float> network { networkTopology };

  common::Mutagen<float> mutagen;
  convolution::Size2D size;

  return 0;
}