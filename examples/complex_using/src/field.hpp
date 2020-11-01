#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>
#include <SFML/Graphics.hpp>

namespace cnn
{
  namespace examples
  {
    namespace complex_using
    {
      template <typename T>
      class Field
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Field();

        size_t GetWidth() const;
        size_t GetHeight() const;

        T GetValue(const size_t x,
                   const size_t y) const;

        // Exception guarantee: strong.
        void LoadFromImage(const std::string& fileName);

      private:

        size_t Width;
        size_t Height;

        std::vector<T> Values;

        size_t ToIndex(const size_t x,
                       const size_t y) const;

      };

      template <typename T>
      Field<T>::Field()
        :
        Width{ 0 },
        Height{ 0 }
      {
      }

      template <typename T>
      size_t Field<T>::GetWidth() const
      {
        return Width;
      }

      template <typename T>
      size_t Field<T>::GetHeight() const
      {
        return Height;
      }

      template <typename T>
      T Field<T>::GetValue(const size_t x,
                           const size_t y) const
      {
        return Values.at(ToIndex(x, y));
      }

      template <typename T>
      void Field<T>::LoadFromImage(const std::string& fileName)
      {
        sf::Image image;
        if (image.loadFromFile(fileName) == false)
        {
          throw std::runtime_error("cnn::examples::complex_using::Field::LoadFromImage(), image.loadFromFile() == false.");
        }

        const size_t width = image.getSize().x;
        const size_t height = image.getSize().y;

        const size_t size = width * height;
        if ((size / width) != height)
        {
          throw std::overflow_error("cnn::examples::complex_using::Field::LoadFromImage(), size has been overflowed.");
        }

        std::vector<T> values;
        values.resize(size);

        for (size_t x = 0; x < width; ++x)
        {
          for (size_t y = 0; y < height; ++y)
          {
            if (image.getPixel(x, y) == sf::Color::Black)
            {
              values.at(ToIndex(x, y)) = static_cast<T>(1L);
            } else {
              values.at(ToIndex(x, y)) = static_cast<T>(0L);
            }
          }
        }

        Width = width;
        Height = height;
        Values.swap(values);
      }

      template <typename T>
      size_t Field<T>::ToIndex(const size_t x,
                               const size_t y) const
      {
        const size_t m = y * Width;
        if ((y != 0) && (Width != 0) && ((m / y) != Width))
        {
          throw std::overflow_error("cnn::examples::complex_using::Field::ToIndex(), m has been overflowed.");
        }
        const size_t index = x + m;
        if (index < x)
        {
          throw std::overflow_error("cnn::examples::complex_using::Field::ToIndex(), index has been overflowed.");
        }
        return index;
      }
    }
  }
}