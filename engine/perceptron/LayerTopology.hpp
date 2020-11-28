#pragma once

#include <cstddef>
#include <cstdint>
#include <istream>
#include <ostream>

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      class LayerTopology
      {
      public:

        LayerTopology(const size_t inputCount = 0,
                      const size_t neuronCount = 0);

        LayerTopology(const LayerTopology& topology) noexcept = default;

        LayerTopology(LayerTopology&& topology) noexcept;

        LayerTopology& operator=(const LayerTopology& toplogy) noexcept = default;

        LayerTopology& operator=(LayerTopology&& topology) noexcept;

        bool operator==(const LayerTopology& topology) const noexcept;

        bool operator!=(const LayerTopology& topology) const noexcept;

        size_t GetInputCount() const noexcept;

        void SetInputCount(const size_t inputCount) noexcept;

        size_t GetNeuronCount() const noexcept;

        void SetNeuronCount(const size_t neuronCount) noexcept;

        void Clear() noexcept;

        // Exception guarantee: base for ostream.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        void Load(std::istream& istream);

      private:

        size_t InputCount;
        size_t NeuronCount;

      };
    }
  }
}
