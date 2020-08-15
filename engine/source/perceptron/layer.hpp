#pragma once

#include "i_layer.hpp"
#include "../common/neuron.hpp"
#include "../common/map.hpp"

#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      template <typename T>
      class Layer : public ILayer<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<Layer<T>>;

        Layer(const size_t inputSize,
              const size_t outputSize);

        size_t GetInputSize() const override;

        const common::IMap<T>& GetInput() const override;
        common::IMap<T>& GetInput() override;

        size_t GetOutputSize() const override;

        const common::IMap<T>& GetOutput() const override;
        common::IMap<T>& GetOutput() override;

        void Process() override;

      private:

        size_t InputSize;
        typename common::IMap<T>::Uptr Input;

        size_t NeuronCount;
        std::unique_ptr<typename common::INeuron<T>::Uptr[]> Neurons;

        size_t OutputSize;
       typename common::IMap<T>::Uptr Output;

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
        /*
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
        */
      }
    }
  }
}
