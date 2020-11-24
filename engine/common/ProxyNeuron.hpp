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

        // Exception guarantee: strong for the neuron.
        T GetInput(const size_t index) const;

        // Exception guarantee: strong for the neuron.
        void SetInput(const size_t index, const T value) const;

        // Exception guarantee: strong for the neuron.
        T GetWeight(const size_t index) const;

        // Exception guarantee: strong for the neuron.
        void SetWeight(const size_t index, const T value) const;

        T GetOutput() const noexcept;

        // Exception guarantee: strong for the neuron.
        void SetOutput(const T value) const;

        void GenerateOutput() const;

        void Clear() const noexcept;

        void ClearInputs() const noexcept;

        void ClearWeights() const noexcept;

        // We expect that the method never throws any exception.
        void FillWeights(ValueGenerator<T>& valueGenerator) const noexcept;

        // We expect that the method never throws any exception.
        void Mutate(Mutagen<T>& mutagen) const noexcept;

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
      void ProxyNeuron<T>::SetInput(const size_t index, const T value) const
      {
        Neuron_.SetInput(index, value);
      }

      template <typename T>
      T ProxyNeuron<T>::GetWeight(const size_t index) const
      {
        return Neuron_.GetWeight(index);
      }

      template <typename T>
      void ProxyNeuron<T>::SetWeight(const size_t index, const T value) const
      {
        Neuron_.SetWeight(index, value);
      }

      template <typename T>
      T ProxyNeuron<T>::GetOutput() const noexcept
      {
        return Neuron_.GetOutput();
      }

      template <typename T>
      void ProxyNeuron<T>::SetOutput(const T value) const
      {
        Neuron_.SetOutput(value);
      }

      template <typename T>
      void ProxyNeuron<T>::GenerateOutput() const
      {
        Neuron_.GenerateOutput();
      }

      template <typename T>
      void ProxyNeuron<T>::Clear() const noexcept
      {
        Neuron_.Clear();
      }

      template <typename T>
      void ProxyNeuron<T>::ClearInputs() const noexcept
      {
        Neuron_.ClearInputs();
      }

      template <typename T>
      void ProxyNeuron<T>::ClearWeights() const noexcept
      {
        Neuron_.ClearWeights();
      }

      template <typename T>
      void ProxyNeuron<T>::FillWeights(ValueGenerator<T>& valueGenerator) const noexcept
      {
        Neuron_.FillWeights(valueGenerator);
      }

      template <typename T>
      void ProxyNeuron<T>::Mutate(Mutagen<T>& mutagen) const noexcept
      {
        Neuron_.Mutate(mutagen);
      }

    }
  }
}