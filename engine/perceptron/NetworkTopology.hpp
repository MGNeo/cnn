#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "LayerTopology.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      // It is only a container type for parameters, so it doesn't validate contained values.
      // Contained values are checked by a type, which takes this type as parameter.
      // For example, convolution::Network2D validates correctness of convolution::Network2DTopology.
      // It is that, because only consumer knows the rules of the validating.
      class NetworkTopology
      {
      public:

        NetworkTopology();

        NetworkTopology(const NetworkTopology& topology) = default;

        NetworkTopology(NetworkTopology&& topology) noexcept = default;

        // Exception guarantee: strong for this.
        NetworkTopology& operator=(const NetworkTopology& topology);

        NetworkTopology& operator=(NetworkTopology&& topology) noexcept = default;

        bool operator==(const NetworkTopology& topology) const noexcept;

        bool operator!=(const NetworkTopology& topology) const noexcept;

        // Exception guarantee: strong for this.
        void PushBack(const LayerTopology& topology);

        size_t GetLayerCount() const noexcept;

        const LayerTopology& GetLayerTopology(const size_t index) const;

        const LayerTopology& GetFirstLayerTopology() const;

        const LayerTopology& GetLastLayerTopology() const;

        // It resets the state to zero.
        void Reset() noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        // It loads full state.
        void Load(std::istream& istream);

      private:

        std::vector<LayerTopology> Topologies;

      };

    }
  }
}

