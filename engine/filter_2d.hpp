#pragma once

#include <stdexcept>

#include "i_filter_2d.hpp"

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

    const size_t Width;
    const size_t Height;
    std::unique_ptr<typename ICore2D<T>::Uptr> Cores;

  };

  template <typename T>
  Filter2D<T>::Filter2D(const size_t c, const size_t w, const size_t h)
    :
    Cores(c)
  {
    if (c == 0)
    {
      throw std::invalid_argument("cnn::Filter2D::Filter2D(), c == 0.");
    }
    if (w == 0)
    {
      throw std::invalid_argument("cnn::Filter2D::Filter2D(), w == 0.");
    }
    if (h == 0)
    {
      throw std::invalid_argument("cnn::Filter2D::Filter2D(), h == 0.");
    }
    for (auto& core : Cores)
    {
      core = std::make_unique<Core2D<T>>(w, h);
    }
  }

  template <typename T>
  size_t Filter2D<T>::GetCount() const
  {
    return Cores.size();
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
    if (index >= Cores.size())
    {
      throw std::range_error("cnn::Filter2D::GetCore() const, index >= Cores.size().");
    }
    return *(Cores[index]);
  }

  template <typename T>
  ICore2D<T>& Filter2D<T>::GetCore(const size_t index)
  {
    if (index >= Cores.size())
    {
      throw std::range_error("cnn::Filter2D::GetCore(), index >= Cores.size().");
    }
    return *(Cores[index]);
  }
}