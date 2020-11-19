#pragma once

#include <memory>
#include <type_traits>
#include <stdexcept>

#include "Core2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      /*
      template <typename T>
      class Filter2D
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
        void ClearInputs() override;
        void ClearWeight() override;
        void ClearOutput() override;

        // The result must not be nullptr.
        typename IFilter2D<T>::Uptr Clone(const bool cloneState) const override;

        Filter2D(const Filter2D<T>& filter2D, const bool cloneState);

        void FillWeights(common::IValueGenerator<T>& valueGenerator) override;

        void Mutate(common::IMutagen<T>& mutagen) override;

        void SetActivationFunctions(const common::IActivationFunction<T>& activationFunction) override;

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
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= CoreCount)
        {
          throw std::range_error("cnn::engine::convolution::Filter2D::GetCore(), index >= CoreCount.");
        }
#endif
        return *(Cores[index]);
      }

      template <typename T>
      const ICore2D<T>& Filter2D<T>::GetCore(const size_t index) const
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= CoreCount)
        {
          throw std::range_error("cnn::engine::convolution::Filter2D::GetCore() const, index >= CoreCount.");
        }
#endif
        return *(Cores[index]);
      }

      template <typename T>
      void Filter2D<T>::Clear()
      {
        for (size_t c = 0; c < CoreCount; ++c)
        {
          Cores[c]->ClearInputs();
          Cores[c]->ClearWeights();
          Cores[c]->ClearOutput();
        }
      }

      template <typename T>
      void Filter2D<T>::ClearInputs()
      {
        for (size_t c = 0; c < CoreCount; ++c)
        {
          Cores[c]->ClearInputs();
        }
      }

      template <typename T>
      void Filter2D<T>::ClearWeight()
      {
        for (size_t c = 0; c < CoreCount; ++c)
        {
          Cores[c]->ClearWeights();
        }
      }

      template <typename T>
      void Filter2D<T>::ClearOutput()
      {
        for (size_t c = 0; c < CoreCount; ++c)
        {
          Cores[c]->ClearOutput();
        }
      }

      // The result must not be nullptr.
      template <typename T>
      typename IFilter2D<T>::Uptr Filter2D<T>::Clone(const bool cloneState) const
      {
        return std::make_unique<Filter2D<T>>(*this, cloneState);
      }

      template <typename T>
      Filter2D<T>::Filter2D(const Filter2D<T>& filter2D, const bool cloneState)
        :
        Width{ filter2D.GetWidth() },
        Height{ filter2D.GetHeight() },
        CoreCount{ filter2D.GetCoreCount() },
        Cores{ std::make_unique<typename ICore2D<T>::Uptr[]>(CoreCount) }
      {
        for (size_t c = 0; c < CoreCount; ++c)
        {
          Cores[c] = filter2D.GetCore(c).Clone(cloneState);
        }
      }

      template <typename T>
      void Filter2D<T>::FillWeights(common::IValueGenerator<T>& valueGenerator)
      {
        for (size_t c = 0; c < CoreCount; ++c)
        {
          Cores[c]->FillWeights(valueGenerator);
        }
      }

      template <typename T>
      void Filter2D<T>::Mutate(common::IMutagen<T>& mutagen)
      {
        for (size_t c = 0; c < GetCoreCount(); ++c)
        {
          Cores[c]->Mutate(mutagen);
        }
      }

      template <typename T>
      void Filter2D<T>::SetActivationFunctions(const common::IActivationFunction<T>& activationFunction)
      {
        for (size_t c = 0; c < GetCoreCount(); ++c)
        {
          Cores[c]->SetActivationFunctions(activationFunction);
        }
      }
      */
    }
  }
}
