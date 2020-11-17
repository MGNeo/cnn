#include "common/Neuron.hpp"
#include "common/ProxyNeuron.hpp"

#include "common/Map.hpp"
#include "common/ProxyMap.hpp"

#include "common/Mutagen.hpp"
#include "convolution/Core2D.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

int main(int argc, char** argv)
{
  cnn::engine::common::Neuron<float> neuron;
  cnn::engine::common::ProxyNeuron<float> proxyNeuron = neuron;

  cnn::engine::common::Map<float> map;
  cnn::engine::common::ProxyMap<float> proxyMap = map;

  cnn::engine::common::Mutagen<float> mutagen;

  cnn::engine::common::Size2D<size_t> size;

  cnn::engine::convolution::Core2D<float> core;

  return 0;
}