#pragma once

#include <memory>
#include <type_traits>

#include "i_core_2d.hpp"
#include "../common/neuron.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Core2D : public ICore2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Core2D(const size_t width, const size_t height);

        size_t GetWidth() const override;
        size_t GetHeight() const override;

        T GetInput(const size_t x, const size_t y) const override;
        void SetInput(const size_t x, const size_t y, const T value) override;

        T GetWeight(const size_t x, const size_t y) const override;
        void SetWeight(const size_t x, const size_t y, const T value) override;

        void Process() override;

        T GetOutput() const override;

        void ClearInputs() override;
        void ClearWeights() override;
        void ClearOutput() override;

        typename ICore2D<T>::Uptr Clone(const bool cloneState) const override;

        Core2D(const Core2D<T>& core2D, const bool cloneState);

        void FillWeights(common::IValueGenerator<T>& valueGenerator) override;

        void CrossFrom(const ICore2D<T>& source1,
                       const ICore2D<T>& source2,
                       common::IBinaryRandomGenerator& binaryRandomGenerator) override;

      private:

        size_t Width;
        size_t Height;
        typename common::INeuron<T>::Uptr Neuron_;

        size_t ToIndex(const size_t x, const size_t y) const;

      };

      template <typename T>
      Core2D<T>::Core2D(const size_t width, const size_t height)
        :
        Width{ width },
        Height{ height }
      {
        if (Width == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Core2D::Core2D(), Width == 0.");
        }
        if (Height == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Core2D::Core2D(), Height == 0.");
        }
        const size_t inputCount = Width * Height;
        if ((inputCount / Width) != Height)
        {
          throw std::overflow_error("cnn::engine::convolution::Core2D::Core2D(), inputCount was overflowed.");
        }
        Neuron_ = std::make_unique<common::Neuron<T>>(inputCount);
        ClearInputs();
        ClearWeights();
        ClearOutput();
      }

      template <typename T>
      size_t Core2D<T>::GetWidth() const
      {
        return Width;
      }

      template <typename T>
      size_t Core2D<T>::GetHeight() const
      {
        return Height;
      }

      template <typename T>
      T Core2D<T>::GetInput(const size_t x, const size_t y) const
      {
        const size_t index = ToIndex(x, y);
        return Neuron_->GetInput(index);
      }

      template <typename T>
      void Core2D<T>::SetInput(const size_t x, const size_t y, const T value)
      {
        const size_t index = ToIndex(x, y);
        Neuron_->SetInput(index, value);
      }

      template <typename T>
      T Core2D<T>::GetWeight(const size_t x, const size_t y) const
      {
        const size_t index = ToIndex(x, y);
        return Neuron_->GetWeight(index);
      }

      template <typename T>
      void Core2D<T>::SetWeight(const size_t x, const size_t y, const T value)
      {
        const size_t index = ToIndex(x, y);
        Neuron_->SetWeight(index, value);
      }

      template <typename T>
      void Core2D<T>::Process()
      {
        Neuron_->Process();
      }

      template <typename T>
      T Core2D<T>::GetOutput() const
      {
        return Neuron_->GetOutput();
      }

      template <typename T>
      size_t Core2D<T>::ToIndex(const size_t x, const size_t y) const
      {
        if (x >= Width)
        {
          throw std::range_error("cnn::engine::convolution::Core2D::ToIndex(), x >= Width.");
        }
        if (y >= Height)
        {
          throw std::range_error("cnn::engine::convolution::Core2D::ToIndex(), y >= Height.");
        }
        return x + y * Width;;
      }

      template <typename T>
      void Core2D<T>::ClearInputs()
      {
        Neuron_->ClearInputs();
      }

      template <typename T>
      void Core2D<T>::ClearWeights()
      {
        Neuron_->ClearWeights();
      }

      template <typename T>
      void Core2D<T>::ClearOutput()
      {
        Neuron_->ClearOutput();
      }

      template <typename T>
      typename ICore2D<T>::Uptr Core2D<T>::Clone(const bool cloneState) const
      {
        return std::make_unique<Core2D<T>>(*this, cloneState);
      }

      template <typename T>
      Core2D<T>::Core2D(const Core2D<T>& core2D, const bool cloneState)
        :
        Width{ core2D.GetWidth() },
        Height{ core2D.GetHeight() },
        Neuron_{ core2D.Neuron_->Clone(cloneState) }
      {
      }

      template <typename T>
      void Core2D<T>::FillWeights(common::IValueGenerator<T>& valueGenerator)
      {
        Neuron_->FillWeights(valueGenerator);
      }

      template <typename T>
      void Core2D<T>::CrossFrom(const ICore2D<T>& source1,
                                const ICore2D<T>& source2,
                                common::IBinaryRandomGenerator& binaryRandomGenerator)
      {
        {
          if (GetWidth() != source1.GetWidth())
          {
            throw std::invalid_argument("cnn::engine::convolution::Core2D::CrossFrom(), GetWidth() != source1.GetWidth().");
          }
          if (GetHeight() != source1.GetHeight())
          {
            throw std::invalid_argument("cnn::engine::convolution::Core2D::CrossFrom(), GetHeight() != source1.GetHeight().");
          }
        }
        {
          if (GetWidth() != source2.GetWidth())
          {
            throw std::invalid_argument("cnn::engine::convolution::Core2D::CrossFrom(), GetWidth() != source2.GetWidth().");
          }
          if (GetHeight() != source2.GetHeight())
          {
            throw std::invalid_argument("cnn::engine::convolution::Core2D::CrossFrom(), GetHeight() != source2.GetHeight().");
          }
        }
        // ...
      }
    }
  }
}