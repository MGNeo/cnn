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
// TODO: Think about helpers for building of topologies.

// VERY IMPORTANT TODO: In order to protect complex object from breaking their consistency, we must add
// RefX and ConstRefX types (where X is name, like Neuron or Map) for every type, which has such characteristic as topology.
// RefX and ConstRefX are wrapper-types, which implement semantics of references, which forbid any methods,
// which can change topology of their target object.
// Complex objects will protect themselves by returning RefX and ConstRefX instead of raw references to their subobjects.
// We need about 20 types for implement this idea.

// TODO: Add pooling handler in layers, which will reduce output size.
// For example 32 x 32 -> 16 x 16.
// It will allow to increase speed of processing very much (from seconds to milliseconds).

using namespace cnn::engine;

int main(int argc, char** argv)
{
  try
  {
    convolution::Network2DTopology convolutionNetwork2DTopology;
    {
      {
        convolution::Layer2DTopology convolutionLayer2DTopology;
        convolutionLayer2DTopology.SetInputSize({ 32, 32 });
        convolutionLayer2DTopology.SetInputCount(3);
        convolutionLayer2DTopology.SetFilterTopology({ { 3, 3 }, 3 });
        convolutionLayer2DTopology.SetFilterCount(15);
        convolutionLayer2DTopology.SetOutputSize({ 30, 30 });
        convolutionLayer2DTopology.SetOutputCount(15);
        convolutionNetwork2DTopology.PushBack(convolutionLayer2DTopology);
      }
      {
        convolution::Layer2DTopology convolutionLayer2DTopology;
        convolutionLayer2DTopology.SetInputSize({ 30, 30 });
        convolutionLayer2DTopology.SetInputCount(15);
        convolutionLayer2DTopology.SetFilterTopology({ { 3, 3 }, 15 });
        convolutionLayer2DTopology.SetFilterCount(5);
        convolutionLayer2DTopology.SetOutputSize({ 28, 28 });
        convolutionLayer2DTopology.SetOutputCount(5);
        convolutionNetwork2DTopology.PushBack(convolutionLayer2DTopology);
      }
      {
        convolution::Layer2DTopology convolutionLayer2DTopology;
        convolutionLayer2DTopology.SetInputSize({ 28, 28 });
        convolutionLayer2DTopology.SetInputCount(5);
        convolutionLayer2DTopology.SetFilterTopology({ { 3, 3 }, 5 });
        convolutionLayer2DTopology.SetFilterCount(6);
        convolutionLayer2DTopology.SetOutputSize({ 26, 26 });
        convolutionLayer2DTopology.SetOutputCount(6);
        convolutionNetwork2DTopology.PushBack(convolutionLayer2DTopology);
      }
    }

    perceptron::NetworkTopology perceptronNetworkTopology;
    {
      {
        perceptron::LayerTopology perceptronLayerTopology;
        perceptronLayerTopology.SetInputCount(convolutionNetwork2DTopology.GetLastLayerTopology().GetOutputValueCount());
        perceptronLayerTopology.SetNeuronCount(15);
        perceptronNetworkTopology.PushBack(perceptronLayerTopology);
      }
      {
        perceptron::LayerTopology perceptronLayerTopology;
        perceptronLayerTopology.SetInputCount(perceptronNetworkTopology.GetLastLayerTopology().GetNeuronCount());
        perceptronLayerTopology.SetNeuronCount(8);
        perceptronNetworkTopology.PushBack(perceptronLayerTopology);
      }
      {
        perceptron::LayerTopology perceptronLayerTopology;
        perceptronLayerTopology.SetInputCount(perceptronNetworkTopology.GetLastLayerTopology().GetNeuronCount());
        perceptronLayerTopology.SetNeuronCount(3);
        perceptronNetworkTopology.PushBack(perceptronLayerTopology);
      }
    }

    complex::Network2DTopology complexNetwork2DTopology;
    complexNetwork2DTopology.SetConvolutionTopology(convolutionNetwork2DTopology);
    complexNetwork2DTopology.SetPerceptronTopology(perceptronNetworkTopology);

    complex::Network2D<float> complexNetwork2D;
    complexNetwork2D.SetTopology(complexNetwork2DTopology);

    {
      std::fstream file("C:/Users/MGNeo/Desktop/ComplexNetwork2D.float", std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);
      complexNetwork2D.Save(file);
    }

    {
      std::fstream file("C:/Users/MGNeo/Desktop/ComplexNetwork2D.float", std::ios_base::in | std::ios_base::binary);
      complexNetwork2D.Load(file);
    }

    complexNetwork2D.GenerateOutput();
  }
  catch (const std::exception& e)
  {
    std::cout << e.what() << std::endl;
  }

  return 0;
}