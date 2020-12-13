#pragma once

#include "../convolution/Size2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      class Lesson2DTopology
      {
      public:

        Lesson2DTopology(const convolution::Size2D& inputSize = {},
                         const size_t neuronCount = {}) noexcept;

        Lesson2DTopology(const Lesson2DTopology& topology) noexcept;

        Lesson2DTopology(Lesson2DTopology&& topology) noexcept;

        Lesson2DTopology& operator=(const Lesson2DTopology& topology) noexcept;

        Lesson2DTopology& operator=(Lesson2DTopology&& topology) noexcept;

        bool operator==(const Lesson2DTopology& topology) const noexcept;

        bool operator!=(const Lesson2DTopology& topology) const noexcept;

        const convolution::Size2D& GetInputSize() const noexcept;

        void SetInputSize(const convolution::Size2D& inputSize) noexcept;

        size_t GetNeuronCount() const noexcept;

        void SetNeuronCount(const size_t neuronCount) noexcept;

        // It resets the state to zero.
        void Reset() noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        // It loads full state.
        void Load(std::istream& istream);

      private:

        convolution::Size2D InputSize;
        size_t NeuronCount;

      };
    }
  }
}