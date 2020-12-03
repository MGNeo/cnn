#pragma once

#include "Neuron.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      // ConstRefNeuron is a wrapper which implements semantics of a safe const reference on Neuron.
      // The safe const reference allows to use methods of the target object, but doesn't allow methods which can change
      // the topology of the target object.
      // This is necessary to protect structure of a complex object which contains the target object as its part.
      template <typename T>
      class ConstRefNeuron
      {
        static_assert(std::is_floating_point<T>::value);

      public:

        ConstRefNeuron(const Neuron<T>& neuron) noexcept;

        ConstRefNeuron(const ConstRefNeuron& constRefNeuron) noexcept;

        ConstRefNeuron(ConstRefNeuron&& constRefNeuron) noexcept = delete;

        ConstRefNeuron& operator=(const ConstRefNeuron& constRefNeuron) noexcept = delete;

        ConstRefNeuron& operator=(ConstRefNeuron&& constRefNeuron) noexcept = delete;

        size_t GetInputCount() const noexcept;

        T GetInput(const size_t index) const;

        T GetWeight(const size_t index) const;

        T GetOutput() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

      private:

        const Neuron<T>& Neuron_;

      };

      template <typename T>
      ConstRefNeuron<T>::ConstRefNeuron(const Neuron<T>& neuron) noexcept
        :
        Neuron_{ neuron }
      {
      }

      template <typename T>
      ConstRefNeuron<T>::ConstRefNeuron(const ConstRefNeuron& constRefNeuron) noexcept
        :
        Neuron_{ constRefNeuron.Neuron_ }
      {
      }

      template <typename T>
      size_t ConstRefNeuron<T>::GetInputCount() const noexcept
      {
        return Neuron_.GetInputCount();
      }

      template <typename T>
      T ConstRefNeuron<T>::GetInput(const size_t index) const
      {
        return Neuron_.GetInput(index);
      }

      template <typename T>
      T ConstRefNeuron<T>::GetWeight(const size_t index) const
      {
        return Neuron_.GetWeight(index);
      }

      template <typename T>
      T ConstRefNeuron<T>::GetOutput() const noexcept
      {
        return Neuron_.GetOutput();
      }

      template <typename T>
      void ConstRefNeuron<T>::Save(std::ostream& ostream) const
      {
        Neuron_.Save(ostream);
      }
    }
  }
}