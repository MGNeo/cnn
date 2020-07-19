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

    Filter2D(const size_t coreCount, const size_t width, const size_t height);

    size_t GetCoreCount() const override;

    size_t GetWidth() const override;
    size_t GetHeight() const override;

    const ICore2D<T>& GetCore(const size_t index) const override;
    ICore2D<T>& GetCore(const size_t index) override;

  private:

    const size_t CoreCount;
    const size_t Width;
    const size_t Height;
    const std::unique_ptr<typename ICore2D<T>::Uptr[]> Cores;

  };

  template <typename T>
  Filter2D<T>::Filter2D(const size_t coreCount, const size_t width, const size_t height)
    :
    CoreCount{ coreCount },
    Width{ width },
    Height{ height },
    Cores{ std::make_unique<typename Core2D<T>::Uptr[]>(CoreCount) }
  {
    if (CoreCount == 0)
    {
      throw std::invalid_argument("cnn::Filter2D::Filter2D(), CoreCount == 0.");
    }
    if (Width == 0)
    {
      throw std::invalid_argument("cnn::Filter2D::Filter2D(), Width == 0.");
    }
    if (Height == 0)
    {
      throw std::invalid_argument("cnn::Filter2D::Filter2D(), Height == 0.");
    }
    for (size_t i = 0; i < CoreCount; ++i)
    {
      Cores[i] = std::make_unique<Core2D<T>>(Width, Height);
    }
  }

  template <typename T>
  size_t Filter2D<T>::GetCoreCount() const
  {
    return CoreCount;
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
    if (index >= CoreCount)
    {
      throw std::range_error("cnn::Filter2D::GetCore() const, index >= CoreCount.");
    }
    return *(Cores[index]);
  }

  template <typename T>
  ICore2D<T>& Filter2D<T>::GetCore(const size_t index)
  {
    if (index >= CoreCount)
    {
      throw std::range_error("cnn::Filter2D::GetCore(), index >= CoreCount.");
    }
    return *(Cores[index]);
  }
}