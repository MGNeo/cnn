#pragma once

#include "Size2D.hpp"
#include "Filter2DTopology.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      class Layer2DTopology
      {
      public:

        Layer2DTopology(const Size2D inputSize = {},
                        const size_t inputCount = 0,
                        const Filter2DTopology filterTopology = {},
                        const size_t filterCount = 0,
                        const Size2D outputSize = {},
                        const size_t outputCount = 0);

        Layer2DTopology(const Layer2DTopology& topology) noexcept = default;

        Layer2DTopology(Layer2DTopology&& topology) noexcept;

        Layer2DTopology& operator=(const Layer2DTopology& topology) noexcept = default;

        Layer2DTopology& operator=(Layer2DTopology&& topology) noexcept;

        Size2D GetInputSize() const noexcept;

        void SetInputSize(const Size2D& inputSize);

        size_t GetInputCount() const noexcept;

        void SetInputCount(const size_t inputCount);

        Filter2DTopology GetFilterTopology() const noexcept;

        void SetFilterTopology(const Filter2DTopology& filterTopology) noexcept;

        size_t GetFilterCount() const noexcept;

        void SetFilterCount(const size_t filterCount) noexcept;

        Size2D GetOutputSize() const noexcept;

        void SetOutputSize(const Size2D& outputSize) noexcept;

        size_t GetOutputCount() const noexcept;

        void SetOutputCount(const size_t outputCount) noexcept;

        void Clear() noexcept;

        // Exception guarantee: base for ostream.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        void Load(std::istream& istream);

      private:

        Size2D InputSize;
        size_t InputCount;

        Filter2DTopology FilterTopology;
        size_t FilterCount;

        Size2D OutputSize;
        size_t OutputCount;

      };
    }
  }
}