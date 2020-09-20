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

        void ClearInputs() override;
        void ClearWeight() override;
        void ClearOutput() override;

        typename INeuron<T>::Uptr Clone(const bool cloneState) const override;

        Neuron(const Neuron<T>& neuron, const bool cloneState);

        void FillWeights(IValueGenerator<T>& valueGenerator) override;

        void CrossFrom(const INeuron<T>& source1,
                       const INeuron<T>& source2) override;

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
        Inputs = std::make_unique<T[]>(InputCount);
        Weights = std::make_unique<T[]>(InputCount);
        ClearInputs();
        ClearWeight();
        ClearOutput();
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
      void Neuron<T>::ClearInputs()
      {
        for (size_t i = 0; i < InputCount; ++i)
        {
          Inputs[i] = 0;
        }
      }

      template <typename T>
      void Neuron<T>::ClearWeight()
      {
        for (size_t w = 0; w < InputCount; ++w)
        {
          Weights[w] = 0;
        }
      }

      template <typename T>
      void Neuron<T>::ClearOutput()
      {
        Output = 0;
      }

      template <typename T>
      typename INeuron<T>::Uptr Neuron<T>::Clone(const bool cloneState) const
      {
        return std::make_unique<Neuron<T>>(*this, cloneState);
      }

      template <typename T>
      Neuron<T>::Neuron(const Neuron<T>& neuron, const bool cloneState)
        :
        InputCount{ neuron.GetInputCount() },
        Inputs{ std::make_unique<T[]>(InputCount) },
        Weights{ std::make_unique<T[]>(InputCount) }
      {
        if (cloneState == true)
        {
          memcpy(Inputs.get(), neuron.Inputs.get(), sizeof(T) * InputCount);
          memcpy(Weights.get(), neuron.Weights.get(), sizeof(T) * InputCount);
          Output = neuron.GetOutput();
        } else {
          for (size_t i = 0; i < InputCount; ++i)
          {
            Inputs[i] = 0;
            Weights[i] = 0;
          }
          Output = 0;
        }
      }

      template <typename T>
      void Neuron<T>::FillWeights(IValueGenerator<T>& valueGenerator)
      {
        for (size_t i = 0; i < InputCount; ++i)
        {
          Weights[i] = valueGenerator.Generate();
        }
      }


      template <typename T>
      void Neuron<T>::CrossFrom(const INeuron<T>& source1,
                                const INeuron<T>& source2)
      {
        // ...
      }
    }
  }
}