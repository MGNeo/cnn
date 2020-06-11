#pragma once

#include <memory>
#include <stdexcept>

namespace cnn
{
  template <typename T>
  class Layer
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Layer(const size_t count);

    size_t GetCount() const;

    T GetCell(const size_t index) const;
    void SetCell(const size_t index, const T value);

  private:

    const size_t Count;
    std::unique_ptr<T[]> Cells;

  };

  template <typename T>
  Layer<T>::Layer(const size_t count)
    :
    Count{ count },
    Cells{ std::make_unique<T[]>(Count) }
  {
    if (Count == 0)
    {
      throw std::invalid_argument("cnn::Layer::Layer(), Count == 0.");
    }
    memset(Cells.get(), 0, Count * sizeof(T));
  }

  template <typename T>
  size_t Layer<T>::GetCount() const
  {
    return Count;
  }

  template <typename T>
  T Layer<T>::GetCell(const size_t index) const
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Layer::GetCell(), index >= Count.");
    }
    return Cells[index];
  }

  template <typename T>
  void Layer<T>::SetCell(const size_t index, const T value)
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Layer::SetCell(), index >= Count.");
    }
    Cells[index] = value;
  }
}
