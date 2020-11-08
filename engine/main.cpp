#include "common/Neuron.hpp"
#include "common/Mutagen.hpp"
#include "convolution/Core2D.hpp"

#include <sstream>
#include <iostream>

// TODO: Core2D.hpp (exception guarantee).

int main(int argc, char** argv)
{
  cnn::engine::common::Neuron<float> neuron;

  cnn::engine::common::Mutagen<float> mutagen;

  cnn::engine::common::Size2D<size_t> size;

  cnn::engine::convolution::Core2D<float> core;

  return 0;
}