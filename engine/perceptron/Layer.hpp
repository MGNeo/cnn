#pragma once

#include "LayerTopology.hpp"
#include "../common/Map.hpp"
#include "../common/Neuron.hpp"

#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      template <typename T>
      class Layer
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Layer(const LayerTopology& topology = {});

        Layer(const Layer& layer);

        Layer(Layer&& layer) noexcept = default;

        Layer& operator=(const Layer& layer);

        Layer& operator=(Layer&& layer) noexcept = default;

        LayerTopology GetTopology() const noexcept;

        // Exception guarantee: strong for this.
        void SetTopology(const LayerTopology& topology);

        const common::Map<T>& GetInput() const noexcept;

        common::Map<T>& GetInput() noexcept;

        const common::Neuron<T>& GetNeuron(const size_t index) const;

        // Exception guarantee: strong for this.
        common::Neuron<T>& GetNeuron(const size_t index);

        const common::Map<T>& GetOutput() const noexcept;

        common::Map<T>& GetOutput() noexcept;

        // Exception guarantee: base for this.
        void GenerateOutput();

        // It clears the state without changing of the topology.
        void Clear() noexcept;

        // It resets the state to zero including the topology.
        void Reset() noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        // It loads full state.
        void Load(std::istream& istream);

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) noexcept;

      private:

        LayerTopology Topology;

        common::Map<T> Input;
        std::unique_ptr<common::Neuron<T>[]> Neurons;
        common::Map<T> Output;

        void CheckTopology(const LayerTopology& topology) const;

      };

      template <typename T>
      Layer<T>::Layer(const LayerTopology& topology)
      {
        CheckTopology(topology);

        Topology = topology;

        Input.SetValueCount(Topology.GetInputCount());

        if (Topology.GetNeuronCount() != 0)
        {
          Neurons = std::make_unique<common::Neuron<T>[]>(Topology.GetNeuronCount());
          for (size_t i = 0; i < Topology.GetNeuronCount(); ++i)
          {
            Neurons[i].SetInputCount(Topology.GetInputCount());
          }
        }
        
        Output.SetValueCount(Topology.GetNeuronCount());
      }

      template <typename T>
      Layer<T>::Layer(const Layer& layer)
        :
        Topology{ layer.Topology },
        Input{ layer.Input },
        Output{ layer.Output }
      {
        if (Topology.GetNeuronCount() != 0)
        {
          Neurons = std::make_unique<common::Neuron<T>[]>(Topology.GetNeuronCount());
          for (size_t i = 0; i < Topology.GetNeuronCount(); ++i)
          {
            Neurons[i] = layer.Neurons[i];
          }
        }
      }

      template <typename T>
      Layer<T>& Layer<T>::operator=(const Layer& layer)
      {
        if (this != &layer)
        {
          Layer<T> tmpLayer{ layer };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpLayer);
        }
        return *this;
      }


      template <typename T>
      LayerTopology Layer<T>::GetTopology() const noexcept
      {
        return Topology;
      }

      template <typename T>
      void Layer<T>::SetTopology(const LayerTopology& topology)
      {
        Layer<T> tmpLayer{ topology };
        // Beware, it is very intimate place for strong exception guarantee.
        std::swap(*this, tmpLayer);
      }

      template <typename T>
      const common::Map<T>& Layer<T>::GetInput() const noexcept
      {
        return Input;
      }

      template <typename T>
      common::Map<T>& Layer<T>::GetInput() noexcept
      {
        return Input;
      }
      
      template <typename T>
      const common::Neuron<T>& Layer<T>::GetNeuron(const size_t index) const
      {
        if (index >= Topology.GetNeuronCount())
        {
          throw std::range_error("cnn::engine::perceptron::Layer::GetNeuron() const, index >= Topology.GetNeuronCount().");
        }
        return Neurons[index];
      }

      template <typename T>
      common::Neuron<T>& Layer<T>::GetNeuron(const size_t index)
      {
        if (index >= Topology.GetNeuronCount())
        {
          throw std::range_error("cnn::engine::perceptron::Layer::GetNeuron(), index >= Topology.GetNeuronCount().");
        }
        return Neurons[index];
      }

      template <typename T>
      const common::Map<T>& Layer<T>::GetOutput() const noexcept
      {
        return Output;
      }

      template <typename T>
      common::Map<T>& Layer<T>::GetOutput() noexcept
      {
        return Output;
      }

      template <typename T>
      void Layer<T>::GenerateOutput()
      {
        for (size_t n = 0; n < Topology.GetNeuronCount(); ++n)
        {
          auto& neuron = Neurons[n];
          for (size_t i = 0; i < Topology.GetInputCount(); ++i)
          {
            neuron.SetInput(i, Input.GetValue(i));
          }
          neuron.GenerateOutput();
          Output.SetValue(n, neuron.GetOutput());
        }
      }

      template <typename T>
      void Layer<T>::Clear() noexcept
      {
        Input->Clear();
        for (size_t i = 0; i < Topology.GetNeuronCount(); ++i)
        {
          Neurons[i].Clear();
        }
        Output.Clear();
      }

      template <typename T>
      void Layer<T>::Reset() noexcept
      {
        Topology.Clear();
        Input.Reset();
        Neurons.reset(nullptr);
        Output.Reset();
      }

      template <typename T>
      void Layer<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::perceptron::Layer::Save(), ostream.good() == false.");
        }

        Topology.Save(ostream);
        Input.Save(ostream);
        for (size_t i = 0; i < Topology.GetNeuronCount(); ++i)
        {
          Neurons[i].Save(ostream);
        }
        Output.Save(ostream);

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::perceptron::Layer::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Layer<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::perceptron::Layer::Load(), istream.good() == false.");
        }

        decltype(Topology) topology;
        decltype(Input) input;
        decltype(Neurons) neurons;
        decltype(Output) output;

        topology.Load(istream);
        CheckTopology(topology);

        input.Load(istream);
        if (input.GetValueCount() != topology.GetInputCount())
        {
          throw std::logic_error("cnn::engine::perceptron::Layer::Load(), input.GetValueCount() != topology.GetInputCount().");
        }

        if (topology.GetNeuronCount() != 0)
        {
          neurons = std::make_unique<common::Neuron<T>[]>(topology.GetNeuronCount());
          for (size_t i = 0; i < topology.GetNeuronCount(); ++i)
          {
            neurons[i].Load(istream);
            if (neurons[i].GetInputCount() != topology.GetInputCount())
            {
              throw std::logic_error("cnn::engine::perceptron::Layer::Load(), neurons[i].GetInputCount() != topology.GetInputCount().");
            }
          }
        }

        output.Load(istream);
        if (output.GetValueCount() != topology.GetNeuronCount())
        {
          throw std::logic_error("cnn::engine::perceptron::Layer::Load(), output.GetValueCount() != topology.GetNeuronCount().");
        }

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::perceptron::Layer::Load(), istream.good() == false.");
        }

        Topology = std::move(topology);
        Input = std::move(input);
        Neurons = std::move(neurons);
        Output = std::move(output);
      }

      template <typename T>
      void Layer<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept
      {
        for (size_t i = 0; i < Topology.GetNeuronCount(); ++i)
        {
          Neurons[i].FillWeights(valueGenerator);
        }
      }

      template <typename T>
      void Layer<T>::Mutate(common::Mutagen<T>& mutagen) noexcept
      {
        for (size_t i = 0; i < Topology.GetNeuronCount(); ++i)
        {
          Neurons[i].Mutate(mutagen);
        }
      }

      template <typename T>
      void Layer<T>::CheckTopology(const LayerTopology& topology) const
      {
        // Zero topology is allowed.
        if ((topology.GetInputCount() == 0) && (topology.GetNeuronCount() == 0))
        {
          return;
        }

        if ((topology.GetInputCount() == 0) || (topology.GetNeuronCount() == 0))
        {
          throw std::invalid_argument("cnn::engine::perceptron::Layer::CheckTopology(), (topology.GetInputCount() == 0) || (topology.GetNeuronCount() == 0).");
        }
      }
    }
  }
}