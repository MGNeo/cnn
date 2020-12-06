#pragma once

#include "Size2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      class Filter2DTopology
      {
      public:

        Filter2DTopology(const Size2D& size = {}, const size_t coreCount = 0) noexcept;

        Filter2DTopology(const Filter2DTopology& topology) noexcept = default;

        Filter2DTopology(Filter2DTopology&& topology) noexcept;

        Filter2DTopology& operator=(const Filter2DTopology& topology) noexcept = default;

        Filter2DTopology& operator=(Filter2DTopology&& topology) noexcept;

        bool operator==(const Filter2DTopology& topology) const noexcept;

        bool operator!=(const Filter2DTopology& topology) const noexcept;

        const Size2D& GetSize() const noexcept;

        void SetSize(const Size2D& size) noexcept;

        size_t GetCoreCount() const noexcept;

        void SetCoreCount(const size_t coreCount) noexcept;

        void Clear() noexcept;

        // Exception guarantee: base for ostream.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        void Load(std::istream& istream);

      private:

        Size2D Size;
        size_t CoreCount;

      };
    }
  }
}