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

        // It resets the state to zero including the topology.
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

