#include "common/Neuron.hpp"
#include "common/Mutagen.hpp"
#include "convolution/Core2D.hpp"
#include "common/Map.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

// TODO: Core2D.hpp (exception guarantee).

int main(int argc, char** argv)
{
  cnn::engine::common::Neuron<float> neuron;

  cnn::engine::common::Mutagen<float> mutagen;

  cnn::engine::common::Size2D<size_t> size;

  cnn::engine::convolution::Core2D<float> core;

  cnn::engine::common::Map<float> map;

  return 0;
}