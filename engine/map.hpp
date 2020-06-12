#pragma once

#include <memory>
#include <stdexcept>

#include "i_map.hpp"

namespace cnn
{
  template <typename T>
  class Map : public IMap<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Map(const size_t count);

    size_t GetCount() const;

    T GetCell(const size_t index) const;
    void SetCell(const size_t index, const T value);

  private:

    const size_t Count;
    std::unique_ptr<T[]> Cells;

  };

  template <typename T>
  Map<T>::Map(const size_t count)
    :
    Count{ count },
    Cells{ std::make_unique<T[]>(Count) }
  {
    if (Count == 0)
    {
      throw std::invalid_argument("cnn::Map::Map(), Count == 0.");
    }
    memset(Cells.get(), 0, Count * sizeof(T));
  }

  template <typename T>
  size_t Map<T>::GetCount() const
  {
    return Count;
  }

  template <typename T>
  T Map<T>::GetCell(const size_t index) const
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Map::GetCell(), index >= Count.");
    }
    return Cells[index];
  }

  template <typename T>
  void Map<T>::SetCell(const size_t index, const T value)
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Map::SetCell(), index >= Count.");
    }
    Cells[index] = value;
  }
}
