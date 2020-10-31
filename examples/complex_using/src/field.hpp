#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cnn
{
  namespace examples
  {
    namespace complex_using
    {
      class Field
      {
      public:

        Field(const size_t width,
              const size_t height);

        size_t GetWidth() const;
        size_t GetHeight() const;

        float GetValue(const size_t x,
                       const size_t y) const;

        void SetValue(const size_t x,
                      const size_t y,
                      const float value);

      private:

        size_t Width;
        size_t Height;

        std::vector<float> Values;

        size_t ToIndex(const size_t x,
                       const size_t y) const;

      };
    }
  }
}