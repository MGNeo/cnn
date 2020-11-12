#include "common/Neuron.hpp"
#include "common/Mutagen.hpp"
#include "convolution/Core2D.hpp"
#include "common/Map.hpp"

#include <sstream>
#include <iostream>
#include <fstream>

// TODO: Add sticky flag "LockedTopology" for any elements which have topology.
// The flag will block changing of topology of an element.
// It is necessary for user can't break topology of sub elements of complex objects through "setters" and "copy/move" operators.
// For example: network.GetLayer(0).GetFilter(0).GetCore(0).SetSize(...);// Oops, topology of network can be broken!

int main(int argc, char** argv)
{
  cnn::engine::common::Neuron<float> neuron;

  cnn::engine::common::Mutagen<float> mutagen;

  cnn::engine::common::Size2D<size_t> size;

  cnn::engine::convolution::Core2D<float> core;

  cnn::engine::common::Map<float> map;

  return 0;
}