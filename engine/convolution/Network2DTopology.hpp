#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>

#include "Layer2DTopology.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      class Network2DTopology
      {
      public:

        Network2DTopology();

        Network2DTopology(const Network2DTopology& topology) = default;

        Network2DTopology(Network2DTopology&& topology) noexcept = default;

        // Exception guarantee: strong for this.
        Network2DTopology& operator=(const Network2DTopology& topology);

        Network2DTopology& operator=(Network2DTopology&& topology) noexcept = default;

        bool operator==(const Network2DTopology& topology) const noexcept;

        bool operator!=(const Network2DTopology& topology) const noexcept;

        // Exception guarantee: strong for this.
        void PushBack(const Layer2DTopology& topology);

        size_t GetLayerCount() const noexcept;

        Layer2DTopology GetLayerTopology(const size_t index) const;

        Layer2DTopology GetFirstLayerTopology() const;

        Layer2DTopology GetLastLayerTopology() const;

        // It resets the state to zero including the topology.
        void Reset() noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        // It loads full state.
        void Load(std::istream& istream);

      private:

        std::vector<Layer2DTopology> Topologies;

      };
    }
  }
}

