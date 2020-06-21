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

    size_t GetCount() const override;

    T GetValue(const size_t index) const override;
    void SetValue(const size_t index, const T value) override;

    void Clear() override;

    void Copy(const IMap<T>& map) override;

  private:

    const size_t Count;
    const std::unique_ptr<T[]> Values;

  };

  template <typename T>
  Map<T>::Map(const size_t count)
    :
    Count{ count },
    Values{ std::make_unique<T[]>(Count) }
  {
    if (Count == 0)
    {
      throw std::invalid_argument("cnn::Map::Map(), Count == 0.");
    }
    Clear();
  }

  template <typename T>
  size_t Map<T>::GetCount() const
  {
    return Count;
  }

  template <typename T>
  T Map<T>::GetValue(const size_t index) const
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Map::GetValue(), index >= Count.");
    }
    return Values[index];
  }

  template <typename T>
  void Map<T>::SetValue(const size_t index, const T value)
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Map::SetValue(), index >= Count.");
    }
    Values[index] = value;
  }

  template <typename T>
  void Map<T>::Clear()
  {
    for (size_t i = 0; i < Count; ++i)
    {
      Values[i] = 0;
    }
  }

  template <typename T>
  void Map<T>::Copy(const IMap<T>& map)
  {
    if (Count != map.GetCount())
    {
      throw std::invalid_argument("cnn::Map<T>::Copy(), Count != map.GetCount()");
    }
    for (size_t i = 0; i < Count; ++i)
    {
      Values[i] = map.GetValue(i);
    }
  }

}
