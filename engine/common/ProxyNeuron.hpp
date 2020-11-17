#pragma once

#include "Neuron.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      // ProxyNeuron is a protecting proxy, which protects a Neuron from changing of topology and other dangerous
      // operations, which can break consistency of complex object, which contains a Neuron as its part.
      template <typename T>
      class ProxyNeuron
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyNeuron(Neuron<T>& neuron);

        ProxyNeuron(const ProxyNeuron<T>& proxyNeuron) noexcept;

        ProxyNeuron(ProxyNeuron<T>&& proxyNeuron) = delete;

        ProxyNeuron& operator=(const ProxyNeuron& proxyNeuron) = delete;

        ProxyNeuron& operator=(ProxyNeuron&& proxyNeuron) = delete;

        size_t GetInputCount() const noexcept;

        // Exception guarantee: strong for this.
        T GetInput(const size_t index) const;

        // Exception guarantee: strong for this.
        void SetInput(const size_t index, const T value);

        // Exception guarantee: strong for this.
        T GetWeight(const size_t index) const;

        // Exception guarantee: strong for this.
        void SetWeight(const size_t index, const T value);

        T GetOutput() const noexcept;

        // Exception guarantee: strong for this.
        void SetOutput(const T value);

        void GenerateOutput() noexcept;

        void Clear() noexcept;

        void ClearInputs() noexcept;

        void ClearWeights() noexcept;

        // We expect that the method never throws any exception.
        void FillWeights(ValueGenerator<T>& valueGenerator) noexcept;

        // We expect that the method never throws any exception.
        void Mutate(Mutagen<T>& mutagen) noexcept;

      private:

        Neuron<T>& Neuron_;

      };

      template <typename T>
      ProxyNeuron<T>::ProxyNeuron(Neuron<T>& neuron)
        :
        Neuron_{ neuron }
      {
      }

      template <typename T>
      ProxyNeuron<T>::ProxyNeuron(const ProxyNeuron<T>& proxyNeuron) noexcept
        :
        Neuron_{ proxyNeuron.Neuron_ }
      {
      }

      template <typename T>
      size_t ProxyNeuron<T>::GetInputCount() const noexcept
      {
        return Neuron_.GetInputCount();
      }

      template <typename T>
      T ProxyNeuron<T>::GetInput(const size_t index) const
      {
        return Neuron_.GetInput(index);
      }

      template <typename T>
      void ProxyNeuron<T>::SetInput(const size_t index, const T value)
      {
        Neuron_.SetInput(index, value);
      }

      template <typename T>
      T ProxyNeuron<T>::GetWeight(const size_t index) const
      {
        return Neuron_.GetWeight(index);
      }

      template <typename T>
      void ProxyNeuron<T>::SetWeight(const size_t index, const T value)
      {
        Neuron_.SetWeight(index, value);
      }

      template <typename T>
      T ProxyNeuron<T>::GetOutput() const noexcept
      {
        return Neuron_.GetOutput();
      }

      template <typename T>
      void ProxyNeuron<T>::SetOutput(const T value)
      {
        Neuron_.SetOutput(value);
      }

      template <typename T>
      void ProxyNeuron<T>::GenerateOutput() noexcept
      {
        Neuron_.GenerateOutput();
      }

      template <typename T>
      void ProxyNeuron<T>::Clear() noexcept
      {
        Neuron_.Clear();
      }

      template <typename T>
      void ProxyNeuron<T>::ClearInputs() noexcept
      {
        Neuron_.ClearInputs();
      }

      template <typename T>
      void ProxyNeuron<T>::ClearWeights() noexcept
      {
        Neuron_.ClearWeights();
      }

      template <typename T>
      void ProxyNeuron<T>::FillWeights(ValueGenerator<T>& valueGenerator) noexcept
      {
        Neuron_.FillWeights(valueGenerator);
      }

      template <typename T>
      void ProxyNeuron<T>::Mutate(Mutagen<T>& mutagen) noexcept
      {
        Neuron_.Mutate(mutagen);
      }

    }
  }
}