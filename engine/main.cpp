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
#include "complex/Network2D.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

// TODO: Perhaps, Neuron must use Map instead of [] for Inputs and Weights.
// TODO: Check that any move-assignment operators reset "from".
// TODO: Topology getters must return const X& instead of X.

// VERY IMPORTANT TODO: In order to protect complex object from breaking their consistency, we must add
// RefX and ConstRefX types (where X is name, like Neuron or Map) for every type, which has such characteristic as topology.
// RefX and ConstRefX are wrapper-types, which implement semantics of references, which forbid any methods,
// which can change topology of their target object.
// Complex objects will protect themselves by returning RefX and ConstRefX instead of raw references to their subobjects.
// We need about 20 types for implement this idea.

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

  convolution::Network2D<float> network2D{ network2DTopology };
  network2D.GenerateOutput();

  perceptron::LayerTopology layerTopology{ 64, 10 };
  perceptron::Layer<float> layer{ layerTopology };

  perceptron::NetworkTopology networkTopology;
  networkTopology.PushBack(layerTopology);

  perceptron::Network<float> network { networkTopology };

  complex::Network2DTopology complexNetwork2DTopology{ network2DTopology, networkTopology };

  complex::Network2D<float> complexNetwork2D{};
  // TODO: Check loading and saving of complexNetwork2D.

  common::Mutagen<float> mutagen;
  convolution::Size2D size;

  return 0;
}