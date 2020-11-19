#include "common/Neuron.hpp"
#include "common/ProxyNeuron.hpp"

#include "common/Map.hpp"
#include "common/ProxyMap.hpp"

#include "convolution/Map2D.hpp"
#include "convolution/ProxyMap2D.hpp"

#include "convolution/Core2D.hpp"
#include "convolution/ProxyCore2D.hpp"

#include "convolution/Filter2DTopology.hpp"

#include "common/Mutagen.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

int main(int argc, char** argv)
{
  cnn::engine::common::Neuron<float> neuron;
  cnn::engine::common::ProxyNeuron<float> proxyNeuron{ neuron };

  cnn::engine::common::Map<float> map;
  cnn::engine::common::ProxyMap<float> proxyMap{ map };

  cnn::engine::convolution::Map2D<float> map2D;
  cnn::engine::convolution::ProxyMap2D<float> proxyMap2D{ map2D };

  cnn::engine::convolution::Core2D<float> core;
  cnn::engine::convolution::ProxyCore2D<float> proxyCore{ core };

  cnn::engine::convolution::Filter2DTopology<size_t> filter2DTopology;

  cnn::engine::common::Mutagen<float> mutagen;
  cnn::engine::convolution::Size2D<size_t> size;

  return 0;
}