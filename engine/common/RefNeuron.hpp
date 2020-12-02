#pragma once

#include "Neuron.hpp"

namespace cnn
{
  namespace engine
    {
    namespace common
    {
      // RefNeuron is a wrapper which implements semantics of a safe reference on Neuron.
      // The safe reference allows to use methods of the target object, but doesn't allow methods which can change
      // the topology of the target object.
      // This is necessary to protect structure of a complex object which contain the target object as its part.
      template <typename T>
      class RefNeuron
      {
        static_assert(std::is_floating_point<T>::value);

      public:

        RefNeuron(Neuron<T>& neuron) noexcept;

        RefNeuron(const RefNeuron& refNeuron) noexcept;

        RefNeuron(RefNeuron&& refNeuron) noexcept = delete;

        RefNeuron& operator=(const RefNeuron& refNeuron) noexcept = delete;

        RefNeuron& operator=(RefNeuron&& refNeuron) noexcept = delete;

        size_t GetInputCount() const noexcept;

        T GetInput(const size_t index) const;

        // Exception guarantee: strong for tne neuron.
        void SetInput(const size_t index, const T value);

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
      RefNeuron<T>::RefNeuron(Neuron<T>& neuron) noexcept
        :
        Neuron_{ neuron }
      {
      }

      template <typename T>
      RefNeuron<T>::RefNeuron(const RefNeuron& refNeuron) noexcept
        :
        Neuron_{ refNeuron.Neuron_ }
      {
      }

      template <typename T>
      size_t RefNeuron<T>::GetInputCount() const noexcept
      {
        return Neuron_.GetInputCount();
      }

      template <typename T>
      T RefNeuron<T>::GetInput(const size_t index) const
      {
        return Neuron_.GetInput(index);
      }

      template <typename T>
      void RefNeuron<T>::SetInput(const size_t index, const T value)
      {
        Neuron_.SetInput(index, value);
      }

      template <typename T>
      T RefNeuron<T>::GetWeight(const size_t index) const
      {
        return Neuron_.GetWeight(index);
      }

      template <typename T>
      void RefNeuron<T>::SetWeight(const size_t index, const T value) const
      {
        Neuron_.SetWeight(index, value);
      }

      template <typename T>
      T RefNeuron<T>::GetOutput() const noexcept
      {
        return Neuron_.GetOutput();
      }

      template <typename T>
      void RefNeuron<T>::SetOutput(const T value) const
      {
        Neuron_.SetOutput(value);
      }

      template <typename T>
      void RefNeuron<T>::GenerateOutput() const noexcept
      {
        Neuron_.GenerateOutput();
      }

      template <typename T>
      void RefNeuron<T>::Clear() const noexcept
      {
        Neuron_.Clear();
      }

      template <typename T>
      void RefNeuron<T>::ClearInputs() const noexcept
      {
        Neuron_.ClearInputs();
      }

      template <typename T>
      void RefNeuron<T>::ClearWeights() const noexcept
      {
        Neuron_.ClearWeights();
      }

      template <typename T>
      void RefNeuron<T>::ClearOutput() const noexcept
      {
        Neuron_.ClearOutput();
      }

      template <typename T>
      void RefNeuron<T>::Save(std::ostream& ostream) const
      {
        Neuron_.Save(ostream);
      }

      template <typename T>
      void RefNeuron<T>::FillWeights(ValueGenerator<T>& valueGenerator) const noexcept
      {
        Neuron_.FillWeights(valueGenerator);
      }

      template <typename T>
      void RefNeuron<T>::Mutate(Mutagen<T>& mutagen) const noexcept
      {
        Neuron_.Mutate(mutagen);
      }
    }
  }
}