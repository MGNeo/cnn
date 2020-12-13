#pragma once

#include "Size2D.hpp"
#include "Filter2DTopology.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // It is only a container type for parameters, so it doesn't validate contained values.
      // Contained values are checked by a type, which takes this type as parameter.
      // For example, convolution::Network2D validates correctness of convolution::Network2DTopology.
      // It is that, because only consumer knows the rules of the validating.
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

        bool operator==(const Layer2DTopology& topology) const noexcept;

        bool operator!=(const Layer2DTopology& topology) const noexcept;

        const Size2D& GetInputSize() const noexcept;

        void SetInputSize(const Size2D& inputSize) noexcept;

        size_t GetInputCount() const noexcept;

        void SetInputCount(const size_t inputCount);

        const Filter2DTopology& GetFilterTopology() const noexcept;

        void SetFilterTopology(const Filter2DTopology& filterTopology) noexcept;

        size_t GetFilterCount() const noexcept;

        void SetFilterCount(const size_t filterCount) noexcept;

        const Size2D& GetOutputSize() const noexcept;

        void SetOutputSize(const Size2D& outputSize) noexcept;

        size_t GetOutputCount() const noexcept;

        void SetOutputCount(const size_t outputCount) noexcept;

        size_t GetOutputValueCount() const;

        void Reset() noexcept;

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