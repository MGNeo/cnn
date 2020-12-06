#pragma once

#include "Neuron.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      // NeuronReference is a type which implements semantics of smart reference to Neuron.
      // The smart reference proxies all methods of Neuron and doesn't allow to use methods, which change
      // the topology of the target neuron.
      // It allow to protect consistency of complex objects, which contain the target neuron as its part.
      template <typename T>
      class NeuronReference
      {

        // ...

      private:

        Neuron<T>& Neuron_;

      };
    }
  }
}