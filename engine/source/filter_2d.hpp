#pragma once

#include <stdexcept>

#include "i_filter_2d.hpp"
#include "core_2d.hpp"

namespace cnn
{
  template <typename T>
  class Filter2D : public IFilter2D<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Filter2D(const size_t c, const size_t w, const size_t h);

    size_t GetCount() const override;

    size_t GetWidth() const override;
    size_t GetHeight() const override;

    const ICore2D<T>& GetCore(const size_t index) const override;
    ICore2D<T>& GetCore(const size_t index) override;

  private:

    const size_t Count;
    const size_t Width;
    const size_t Height;
    const std::unique_ptr<typename ICore2D<T>::Uptr[]> Cores;

  };

  template <typename T>
  Filter2D<T>::Filter2D(const size_t c, const size_t w, const size_t h)
    :
    Count{ c },
    Width{ w },
    Height{ h },
    Cores{ std::make_unique<typename Core2D<T>::Uptr[]>(Count) }
  {
    if (Count == 0)
    {
      throw std::invalid_argument("cnn::Filter2D::Filter2D(), Count == 0.");
    }
    if (Width == 0)
    {
      throw std::invalid_argument("cnn::Filter2D::Filter2D(), Width == 0.");
    }
    if (Height == 0)
    {
      throw std::invalid_argument("cnn::Filter2D::Filter2D(), Height == 0.");
    }
    for (size_t i = 0; i < Count; ++i)
    {
      Cores[i] = std::make_unique<Core2D<T>>(w, h);
    }
  }

  template <typename T>
  size_t Filter2D<T>::GetCount() const
  {
    return Count;
  }

  template <typename T>
  size_t Filter2D<T>::GetWidth() const
  {
    return Width;
  }

  template <typename T>
  size_t Filter2D<T>::GetHeight() const
  {
    return Height;
  }

  template <typename T>
  const ICore2D<T>& Filter2D<T>::GetCore(const size_t index) const
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Filter2D::GetCore() const, index >= Count.");
    }
    return *(Cores[index]);
  }

  template <typename T>
  ICore2D<T>& Filter2D<T>::GetCore(const size_t index)
  {
    if (index >= Count)
    {
      throw std::range_error("cnn::Filter2D::GetCore(), index >= Count.");
    }
    return *(Cores[index]);
  }
}