#pragma once

#include <memory>
#include <type_traits>
#include <stdexcept>

#include "i_neuron.hpp"
#include "activation_function.hpp"

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
        void ClearWeights() override;
        void ClearOutput() override;

        // The result must not be nullptr.
        typename INeuron<T>::Uptr Clone(const bool cloneState) const override;

        Neuron(const Neuron<T>& neuron, const bool cloneState);

        void FillWeights(IValueGenerator<T>& valueGenerator) override;

        void CrossFrom(const INeuron<T>& source1,
                       const INeuron<T>& source2,
                       IBinaryRandomGenerator& binaryRandomGenerator) override;

        void Mutate(common::IMutagen<T>& mutagen) override;

        const IActivationFunction<T>& GetActivationFunction() const override;
        void SetActivationFunction(const IActivationFunction<T>& activationFunction) override;

      public:

        size_t InputCount;

        std::unique_ptr<T[]> Inputs;
        std::unique_ptr<T[]> Weights;

        T Output;

        // ActivationFunction_ must not be nullptr.
        typename IActivationFunction<T>::Uptr ActivationFunction_;

      };

      template <typename T>
      Neuron<T>::Neuron(const size_t inputCount)
        :
        InputCount{ inputCount },
        ActivationFunction_{ std::make_unique<common::ActivationFunction<T>>() }
      {
        if (InputCount == 0)
        {
          throw std::invalid_argument("cnn::engine::common::Neuron::Neuron(), InputCount == 0.");
        }
        Inputs = std::make_unique<T[]>(InputCount);
        Weights = std::make_unique<T[]>(InputCount);
        ClearInputs();
        ClearWeights();
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
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::GetInput(), index >= InputCount.");
        }
#endif
        return Inputs[index];
      }

      template <typename T>
      void Neuron<T>::SetInput(const size_t index, const T value)
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::SetInput(), index >= InputCount.");
        }
#endif
        Inputs[index] = value;
      }

      template <typename T>
      T Neuron<T>::GetWeight(const size_t index) const
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::GetWeight(), index >= InputCount.");
        }
#endif
        return Weights[index];
      }

      template <typename T>
      void Neuron<T>::SetWeight(const size_t index, const T value)
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::SetWeight(), index >= InputCount.");
        }
#endif
        Weights[index] = value;
      }

      template <typename T>
      void Neuron<T>::Process()
      {
        Output = 0;
        for (size_t i = 0; i < InputCount; ++i)
        {
          auto a = Inputs[i];
          auto b = Weights[i];
          Output += Inputs[i] * Weights[i];
        }
        // ActivationFunction_ must not be nullptr.
        Output = ActivationFunction_->Use(Output);
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
      void Neuron<T>::ClearWeights()
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

      // The result must not be nullptr.
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
        Weights{ std::make_unique<T[]>(InputCount) },
        ActivationFunction_{ neuron.GetActivationFunction().Clone() }
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
                                const INeuron<T>& source2,
                                IBinaryRandomGenerator& binaryRandomGenerator)
      {
        if (GetInputCount() != source1.GetInputCount())
        {
          throw std::invalid_argument("cnn::engine::common::Neuron::CrossFrom(), GetInputCount() != source1.GetInputCount().");
        }
        if (GetInputCount() != source2.GetInputCount())
        {
          throw std::invalid_argument("cnn::engine::common::Neuron::CrossFrom(), GetInputCount() != source2.GetInputCount().");
        }
        for (size_t w = 0; w < GetInputCount(); ++w)
        {
          const T value = binaryRandomGenerator.Generate() ? source1.GetWeight(w) : source2.GetWeight(w);
          SetWeight(w, value);
        }
      }

      template <typename T>
      void Neuron<T>::Mutate(common::IMutagen<T>& mutagen)
      {
        for (size_t w = 0; w < GetInputCount(); ++w)
        {
          Weights[w] = mutagen.Mutate(Weights[w]);
        }
      }

      template <typename T>
      const IActivationFunction<T>& Neuron<T>::GetActivationFunction() const
      {
        return *ActivationFunction_;
      }

      // The result must not be nullptr.
      template <typename T>
      void Neuron<T>::SetActivationFunction(const IActivationFunction<T>& activationFunction)
      {
        ActivationFunction_ = activationFunction.Clone();
      }
    }
  }
}