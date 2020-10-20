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

        // TODO: Perhaps, default constructor will be more flexible.
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

        // The result must not be nullptr.
        typename ICore2D<T>::Uptr Clone(const bool cloneState) const override;

        Core2D(const Core2D<T>& core2D, const bool cloneState);

        void FillWeights(common::IValueGenerator<T>& valueGenerator) override;

        void Mutate(common::IMutagen<T>& mutagen) override;

        void SetActivationFunctions(const common::IActivationFunction<T>& activationFunction) override;

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
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (x >= Width)
        {
          throw std::range_error("cnn::engine::convolution::Core2D::ToIndex(), x >= Width.");
        }
        if (y >= Height)
        {
          throw std::range_error("cnn::engine::convolution::Core2D::ToIndex(), y >= Height.");
        }
#endif
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

      // The result must not be nullptr.
      template <typename T>
      typename ICore2D<T>::Uptr Core2D<T>::Clone(const bool cloneState) const
      {
        return std::make_unique<Core2D<T>>(*this, cloneState);
      }

      // The result must not be nullptr.
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
      void Core2D<T>::Mutate(common::IMutagen<T>& mutagen)
      {
        Neuron_->Mutate(mutagen);
      }

      template <typename T>
      void Core2D<T>::SetActivationFunctions(const common::IActivationFunction<T>& activationFunction)
      {
        Neuron_->SetActivationFunction(activationFunction);
      }

    }
  }
}