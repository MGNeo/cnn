#pragma once

#include <memory>
#include <type_traits>
#include <stdexcept>

#include "i_filter_2d.hpp"
#include "core_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Filter2D : public IFilter2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Filter2D(const size_t width, const size_t height, const size_t coreCount);

        size_t GetWidth() const override;
        size_t GetHeight() const override;

        size_t GetCoreCount() const override;

        ICore2D<T>& GetCore(const size_t index) override;
        const ICore2D<T>& GetCore(const size_t index) const override;

        void Clear() override;

      private:

        size_t Width;
        size_t Height;
        size_t CoreCount;
        std::unique_ptr<typename ICore2D<T>::Uptr[]> Cores;

      };

      template <typename T>
      Filter2D<T>::Filter2D(const size_t width, const size_t height, const size_t coreCount)
        :
        Width{ width },
        Height{ height },
        CoreCount{ coreCount }
      {
        if (Width == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Filter2D::Filter2D(), Width == 0.");
        }
        if (Height == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Filter2D::Filter2D(), Height == 0.");
        }
        if (CoreCount == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Filter2D::Filter2D(), CoreCount == 0.");
        }
        Cores = std::make_unique<typename ICore2D<T>::Uptr[]>(CoreCount);
        for (size_t i = 0; i < CoreCount; ++i)
        {
          Cores[i] = std::make_unique<Core2D<T>>(Width, Height);
        }
        Clear();
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
      size_t Filter2D<T>::GetCoreCount() const
      {
        return CoreCount;
      }

      template <typename T>
      ICore2D<T>& Filter2D<T>::GetCore(const size_t index)
      {
        if (index >= CoreCount)
        {
          throw std::range_error("cnn::engine::convolution::Filter2D::GetCore(), index >= CoreCount.");
        }
        return *(Cores[index]);
      }

      template <typename T>
      const ICore2D<T>& Filter2D<T>::GetCore(const size_t index) const
      {
        if (index >= CoreCount)
        {
          throw std::range_error("cnn::engine::convolution::Filter2D::GetCore() const, index >= CoreCount.");
        }
        return *(Cores[index]);
      }

      template <typename T>
      void Filter2D<T>::Clear()
      {
        for (size_t c = 0; c < CoreCount; ++c)
        {
          Cores[c]->Clear();
        }
      }

    }
  }
}