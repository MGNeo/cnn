#pragma once

#include <memory>
#include <type_traits>
#include <stdexcept>

#include "i_neuron.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class Neuron : public INeuron<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Neuron(const size_t inputCount);

        size_t GetInputCount() const override;

        T GetInput(const size_t index) const override;
        void SetInput(const size_t index, const T value) override;

        T GetWeight(const size_t index) const override;
        void SetWeight(const size_t index, const T value) override;

        void Process() override;

        T GetOutput() const override;

        void Clear() override;;

      public:

        size_t InputCount;

        std::unique_ptr<T[]> Inputs;
        std::unique_ptr<T[]> Weights;

        T Output;

      };

      template <typename T>
      Neuron<T>::Neuron(const size_t inputCount)
        :
        InputCount{ inputCount }
      {
        if (InputCount == 0)
        {
          throw std::invalid_argument("cnn::engine::common::Neuron::Neuron(), InputCount == 0.");
        }
        Inputs = std::unique_ptr<T[]>(InputCount);
        Weights = std::unique_ptr<T[]>(InputCount);
        Clear();
      }

      template <typename T>
      size_t Neuron<T>::GetInputCount() const
      {
        return InputCount;
      }

      template <typename T>
      T Neuron<T>::GetInput(const size_t index) const
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::GetInput(), index >= InputCount.");
        }
        return Inputs[index];
      }

      template <typename T>
      void Neuron<T>::SetInput(const size_t index, const T value)
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::SetInput(), index >= InputCount.");
        }
        Inputs[index] = value;
      }

      template <typename T>
      T Neuron<T>::GetWeight(const size_t index) const
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::GetWeight(), index >= InputCount.");
        }
        return Weights[index];
      }

      template <typename T>
      void Neuron<T>::SetWeight(const size_t index, const T value)
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::SetWeight(), index >= InputCount.");
        }
        Weights[index] = value;
      }

      template <typename T>
      void Neuron<T>::Process()
      {
        Output = 0;
        for (size_t i = 0; i < InputCount; ++i)
        {
          Output += Inputs[i] * Weights[i];
        }
      }

      template <typename T>
      T Neuron<T>::GetOutput() const
      {
        return Output;
      }

      template <typename T>
      void Neuron<T>::Clear()
      {
        for (size_t i = 0; i < InputCount; ++i)
        {
          Inputs[i] = 0;
          Weights[i] = 0;
        }
      }
    }
  }
}