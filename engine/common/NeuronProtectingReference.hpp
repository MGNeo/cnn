#pragma once

#include "Neuron.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      // NeuronProtectingReference is a type which implements semantics of protecting reference to Neuron.
      // The protecting reference proxies all methods of Neuron and doesn't allow to use methods, which change
      // the topology of the target neuron.
      // It allow to protect consistency of complex objects, which contain the target neuron as its part.
      template <typename T>
      class NeuronProtectingReference
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        NeuronProtectingReference(Neuron<T>& neuron);

        NeuronProtectingReference(const NeuronProtectingReference& NeuronProtectingReference) noexcept;

        NeuronProtectingReference(NeuronProtectingReference&& NeuronProtectingReference) noexcept = delete;

        NeuronProtectingReference& operator=(const NeuronProtectingReference& meuronReference) noexcept = delete;

        NeuronProtectingReference& operator=(NeuronProtectingReference&& NeuronProtectingReference) noexcept = delete;

        size_t GetInputCount() const noexcept;

        T GetInput(const size_t index) const;

        // Exception guarantee: strong for the neuron.
        void SetInput(const size_t index, const T value) const;

        T GetWeight(const size_t index) const;

        // Exception guarantee: strong for the neuron.
        void SetWeight(const size_t index, const T value) const;

        T GetOutput() const noexcept;

        // Exception guarantee: strong for the neuron.
        void SetOutput(const T value) const;

        void GenerateOutput() const noexcept;

        // It clears the state without changing of the topology of the neuron.
        void Clear() const noexcept;

        // It clears the state without changing of the topology of the neuron.
        void ClearInputs() const noexcept;

        // It clears the state without changing of the topology of the neuron.
        void ClearWeights() const noexcept;

        // It clears the state without changing of the topology of the neuron.
        void ClearOutput() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // We expect that the method never throws any exception.
        void FillWeights(ValueGenerator<T>& valueGenerator) const noexcept;

        // We expect that the method never throws any exception.
        void Mutate(Mutagen<T>& mutagen) const noexcept;

      private:

        Neuron<T>& Neuron_;

      };

      template <typename T>
      NeuronProtectingReference<T>::NeuronProtectingReference(Neuron<T>& neuron)
        :
        Neuron_{ neuron }
      {
      }

      template <typename T>
      NeuronProtectingReference<T>::NeuronProtectingReference(const NeuronProtectingReference& NeuronProtectingReference) noexcept
        :
        Neuron_{ NeuronProtectingReference.Neuron_ }
      {
      }

      template <typename T>
      size_t NeuronProtectingReference<T>::GetInputCount() const noexcept
      {
        return Neuron_.GetInputCount();
      }

      template <typename T>
      T NeuronProtectingReference<T>::GetInput(const size_t index) const
      {
        return Neuron_.GetInput(index);
      }

      template <typename T>
      void NeuronProtectingReference<T>::SetInput(const size_t index, const T value) const
      {
        Neuron_.SetInput(index, value);
      }

      template <typename T>
      T NeuronProtectingReference<T>::GetWeight(const size_t index) const
      {
        return Neuron_.GetWeight(index);
      }

      template <typename T>
      void NeuronProtectingReference<T>::SetWeight(const size_t index, const T value) const
      {
        Neuron_.SetWeight(index, value);
      }

      template <typename T>
      T NeuronProtectingReference<T>::GetOutput() const noexcept
      {
        return Neuron_.GetOutput();
      }

      template <typename T>
      void NeuronProtectingReference<T>::SetOutput(const T value) const
      {
        Neuron_.SetOutput(value);
      }

      template <typename T>
      void NeuronProtectingReference<T>::GenerateOutput() const noexcept
      {
        Neuron_.GenerateOutput();
      }

      template <typename T>
      void NeuronProtectingReference<T>::Clear() const noexcept
      {
        Neuron_.Clear();
      }

      template <typename T>
      void NeuronProtectingReference<T>::ClearInputs() const noexcept
      {
        Neuron_.ClearInputs();
      }

      template <typename T>
      void NeuronProtectingReference<T>::ClearWeights() const noexcept
      {
        Neuron_.ClearWeights();
      }

      template <typename T>
      void NeuronProtectingReference<T>::ClearOutput() const noexcept
      {
        Neuron_.ClearOutput();
      }

      template <typename T>
      void NeuronProtectingReference<T>::Save(std::ostream& ostream) const
      {
        Neuron_.Save(ostream);
      }

      template <typename T>
      void NeuronProtectingReference<T>::FillWeights(ValueGenerator<T>& valueGenerator) const noexcept
      {
        Neuron_.FillWeights(valueGenerator);
      }

      template <typename T>
      void NeuronProtectingReference<T>::Mutate(Mutagen<T>& mutagen) const noexcept
      {
        Neuron_.Mutate(mutagen);
      }
    }
  }
}