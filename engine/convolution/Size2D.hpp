#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <istream>
#include <ostream>


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
      class Size2D
      {
      public:
        
        Size2D(const size_t width = 0, const size_t height = 0) noexcept;

        Size2D(const Size2D& size) noexcept = default;

        Size2D(Size2D&& size) noexcept;

        Size2D& operator=(const Size2D& size) noexcept = default;

        Size2D& operator=(Size2D&& size) noexcept;

        bool operator==(const Size2D& size) const noexcept;

        bool operator!=(const Size2D& size) const noexcept;

        size_t GetWidth() const noexcept;

        void SetWidth(const size_t width) noexcept;

        size_t GetHeight() const noexcept;

        void SetHeight(const size_t height) noexcept;

        void Clear() noexcept;

        // Exception guarantee: base for ostream.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        void Load(std::istream& istream);

        size_t GetArea() const;

      private:

        size_t Width;
        size_t Height;

      };
    }
  }
}

