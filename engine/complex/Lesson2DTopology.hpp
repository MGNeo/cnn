#pragma once

#include "../convolution/Size2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      // It is only a container type for parameters, so it doesn't validate contained values.
      // Contained values are checked by a type, which takes this type as parameter.
      // For example, complex::Lesson2D validates correctness of complex::Lesson2DTopology.
      // It is that, because only consumer knows the rules of the validating.
      class Lesson2DTopology
      {
      public:

        Lesson2DTopology(const convolution::Size2D& inputSize = {},
                         const size_t outputSize = {}) noexcept;

        Lesson2DTopology(const Lesson2DTopology& topology) noexcept;

        Lesson2DTopology(Lesson2DTopology&& topology) noexcept;

        Lesson2DTopology& operator=(const Lesson2DTopology& topology) noexcept;

        Lesson2DTopology& operator=(Lesson2DTopology&& topology) noexcept;

        bool operator==(const Lesson2DTopology& topology) const noexcept;

        bool operator!=(const Lesson2DTopology& topology) const noexcept;

        const convolution::Size2D& GetInputSize() const noexcept;

        void SetInputSize(const convolution::Size2D& inputSize) noexcept;

        size_t GetOutputCount() const noexcept;

        void SetOutputCount(const size_t outputCount) noexcept;

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
        size_t OutputCount;

      };
    }
  }
}