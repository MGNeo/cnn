#pragma once

#include <cstdint>
#include <cstddef>
#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    template<typename T, size_t C>
    class Layer
    {

      static_assert(std::is_floating_point<T>::value);
      static_assert(C > 0);

    public:

      Layer();

      T GetValue(const size_t index) const;
      void SetValue(const size_t index, const T value);

      size_t GetCount() const;

    private:

      std::array<T, C> Values;

    };

    template<typename T, size_t C>
    Layer<T, C>::Layer()
      :
      Values{}
    {
    }

    template<typename T, size_t C>
    T Layer<T, C>::GetValue(const size_t index) const
    {
      if (index >= C)
      {
        throw std::range_error("cnn::engine::Layer::GetValue(), index >= C.");
      }
      return Values[index];
    }

    template<typename T, size_t C>
    void Layer<T, C>::SetValue(const size_t index, const T value)
    {
      // TODO: What about controlling of NaN and etc?
      if (index >= C)
      {
        throw std::range_error("cnn::engine::Layer::SetValue(), index >= C.");
      }
      Values[index] = value;
    }

    template<typename T, size_t C>
    size_t Layer<T, C>::GetCount() const
    {
      return C;
    }
  }
}
