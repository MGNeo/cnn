#pragma once

#include "../common/Neuron.hpp"
#include "../common/Map.hpp"

#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      /*
      template <typename T>
      class Layer
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Layer(const size_t inputCount = 0, const size_t neuronCount = 0);

        Layer(const Layer& layer) = default;

        Layer(Layer&& layer) noexcept;

        Layer& operator=(const Layer& layer) = default;

        Layer& operator=(Layer&& layer) noexcept;

        size_t GetInputSize() const noexcept;

        void SetInputSize(const size_t inputSize);

        T GetInputValue(const size_t index);

        void SetInputValue(const size_t index, const T value);

        size_t GetOutputSize() const noexcept;

        void SetOutputSize(const size_t outputSize);

        T GetOutputValue(const size_t index);

        void SetOutputValue(const size_t index, const T value);

        size_t GetNeuronCount() const noexcept;

        void SetNeuronCount(const size_t neuronCount);

        // ...

        void GenerateOutput();

        void FillWeights(common::ValueGenerator<T>& valueGenerator);

        void Mutate(common::Mutagen<T>& mutagen);

      private:

        common::Map<T> Input;

        size_t NeuronCount;
        std::unique_ptr<common::Neuron<T>[]> Neurons;

        common::Map<T> Output;

        // TODO: Clear() family functions.

      };

      template <typename T>
      Layer<T>::Layer(const size_t inputSize,
        const size_t outputSize)
        :
        InputSize{ inputSize },
        NeuronCount{ outputSize },
        OutputSize{ outputSize }
      {
        if (InputSize == 0)
        {
          throw std::invalid_argument("cnn::engine::perceptron::Layer::Layer(const size_t, const size_t), InputSize == 0.");
        }
        if (NeuronCount == 0)
        {
          throw std::invalid_argument("cnn::engine::perceptron::Layer::Layer(const size_t, const size_t), NeuronCount == 0.");
        }
        if (OutputSize == 0)
        {
          throw std::invalid_argument("cnn::engine::perceptron::Layer::Layer(const size_t, const size_t), OutputSize == 0.");
        }

        Input = std::make_unique<common::Map<T>>(InputSize);

        Neurons = std::make_unique<typename common::Neuron<T>::Uptr[]>(NeuronCount);
        for (size_t n = 0; n < NeuronCount; ++n)
        {
          Neurons[n] = std::make_unique<common::Neuron<T>>(InputSize);
        }

        Output = std::make_unique<common::Map<T>>(OutputSize);
      }

      template <typename T>
      size_t Layer<T>::GetInputSize() const
      {
        return InputSize;
      }

      template <typename T>
      const common::IMap<T>& Layer<T>::GetInput() const
      {
        return *(Input);
      }

      template <typename T>
      common::IMap<T>& Layer<T>::GetInput()
      {
        return *(Input);
      }

      template <typename T>
      size_t Layer<T>::GetNeuronCount() const
      {
        return NeuronCount;
      }

      template <typename T>
      const common::INeuron<T>& Layer<T>::GetNeuron(const size_t index) const
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= NeuronCount)
        {
          throw std::range_error("cnn::engine::perceptron::Layer::GetNeuron() const, index >= NeuronCount.");
        }
#endif
        return *(Neurons[index]);
      }

      template <typename T>
      common::INeuron<T>& Layer<T>::GetNeuron(const size_t index)
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= NeuronCount)
        {
          throw std::range_error("cnn::engine::perceptron::Layer::GetNeuron(), index >= NeuronCount.");
        }
#endif
        return *(Neurons[index]);
      }

      template <typename T>
      size_t Layer<T>::GetOutputSize() const
      {
        return OutputSize;
      }

      template <typename T>
      const common::IMap<T>& Layer<T>::GetOutput() const
      {
        return *(Output);
      }

      template <typename T>
      common::IMap<T>& Layer<T>::GetOutput()
      {
        return *(Output);
      }

      template <typename T>
      void Layer<T>::Process()
      {
        // TODO: Think about exception safety.
        for (size_t n = 0; n < NeuronCount; ++n)
        {
          auto& neuron = *(Neurons[n]);
          for (size_t i = 0; i < InputSize; ++i)
          {
            const T value = Input->GetValue(i);
            neuron.SetInput(i, value);
          }
          neuron.Process();
          const T value = neuron.GetOutput();
          Output->SetValue(n, value);

        }
      }

      // The result must not be nullptr.
      template <typename T>
      typename ILayer<T>::Uptr Layer<T>::Clone(const bool cloneState) const
      {
        return std::make_unique<Layer<T>>(*this, cloneState);
      }

      template <typename T>
      Layer<T>::Layer(const Layer<T>& layer, const bool cloneState)
        :
        InputSize{ layer.GetInputSize() },
        Input{ layer.GetInput().Clone(cloneState) },
        NeuronCount{ layer.GetNeuronCount() },
        Neurons{ std::make_unique<typename common::INeuron<T>::Uptr[]>(NeuronCount) },
        OutputSize{ layer.GetOutputSize() },
        Output{ layer.GetOutput().Clone(cloneState) }
      {
        for (size_t n = 0; n < NeuronCount; ++n)
        {
          Neurons[n] = layer.GetNeuron(n).Clone(cloneState);
        }
      }

      template <typename T>
      void Layer<T>::FillWeights(common::IValueGenerator<T>& valueGenerator)
      {
        for (size_t n = 0; n < NeuronCount; ++n)
        {
          Neurons[n]->FillWeights(valueGenerator);
        }
      }

      template <typename T>
      void Layer<T>::Mutate(common::IMutagen<T>& mutagen)
      {
        for (size_t n = 0; n < GetNeuronCount(); ++n)
        {
          Neurons[n]->Mutate(mutagen);
        }
      }

      template <typename T>
      void Layer<T>::SetActivationFunctions(const common::IActivationFunction<T>& activationFunction)
      {
        for (size_t n = 0; n < GetNeuronCount(); ++n)
        {
          Neurons[n]->SetActivationFunction(activationFunction);
        }
      }

      */
    }
  }
}