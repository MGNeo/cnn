#include "common/Neuron.hpp"
#include "common/ProxyNeuron.hpp"

#include "common/Map.hpp"
#include "common/ProxyMap.hpp"

#include "convolution/Map2D.hpp"
#include "convolution/ProxyMap2D.hpp"

#include "convolution/Core2D.hpp"
#include "convolution/ProxyCore2D.hpp"

#include "convolution/Filter2DTopology.hpp"
#include "convolution/Filter2D.hpp"
#include "convolution/ProxyFilter2D.hpp"

#include "convolution/Layer2DTopology.hpp"
#include "convolution/Layer2D.hpp"
#include "convolution/ProxyLayer2D.hpp"

#include "convolution/Network2DTopology.hpp"
#include "convolution/Network2D.hpp"

#include "common/Mutagen.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

// TODO: All methods of proxies must be const.
// TODO: Check that any move-assignment operators reset "from".

using namespace cnn::engine;

int main(int argc, char** argv)
{
  common::Neuron<float> neuron;
  common::ProxyNeuron<float> proxyNeuron{ neuron };

  common::Map<float> map;
  common::ProxyMap<float> proxyMap{ map };

  convolution::Map2D<float> map2D;
  convolution::ProxyMap2D<float> proxyMap2D{ map2D };

  convolution::Core2D<float> core;
  convolution::ProxyCore2D<float> proxyCore{ core };

  convolution::Filter2DTopology filterTopology{ {10, 10}, 10 };
  convolution::Filter2D<float> filter{ filterTopology };
  convolution::ProxyFilter2D<float> proxyFilter{ filter };

  convolution::Layer2DTopology layer2DTopology{ {10, 10}, 10, { {3, 3}, 10 }, 5, {8, 8}, 5 };
  convolution::Layer2D<float> layer2D{ layer2DTopology };
  convolution::ProxyLayer2D<float> proxyLayer2D{ layer2D };

  convolution::Network2DTopology network2DTopology;
  network2DTopology.PushBack(layer2DTopology);
  network2DTopology.PushBack(layer2DTopology);
  network2DTopology.Reset();

  convolution::Network2D<float> network2D{ network2DTopology };

  network2D.GenerateOputput();

  common::Mutagen<float> mutagen;
  convolution::Size2D size;

  return 0;
}