#pragma once

#include "NetworkTopology.hpp"

#include "Layer.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      template <typename T>
      class Network
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Network(const NetworkTopology& topology = {});

        Network(const Network& network);

        Network(Network&& network) noexcept = default;

        Network& operator=(const Network& network);

        Network& operator=(Network&& network) noexcept = default;

        NetworkTopology GetTopology() const;

        // Exception guarantee: base for the network.
        void SetTopology(const NetworkTopology& topology);

        const Layer<T>& GetLayer(const size_t index) const;

        // Exception guarantee: strong for this.
        Layer<T> GetLayer(const size_t index);

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

        NetworkTopology Topology;
        std::unique_ptr<Layer<T>[]> Layers;

        void CheckTopology(const NetworkTopology& topology) const;

      };

      template <typename T>
      Network<T>::Network(const NetworkTopology& topology)
      {
        CheckTopology(topology);

        Topology = topology;

        if (Topology.GetLayerCount() != 0)
        {
          Layers = std::make_unique<Layer<T>[]>(Topology.GetLayerCount());
          for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
          {
            Layers[i].SetTopology(Topology.GetLayerTopology(i));
          }
        }
      }

      template <typename T>
      Network<T>::Network(const Network& network)
        :
        Topology{ network.Topology }
      {
        if (Topology.GetLayerCount() != 0)
        {
          Network<T> tmpNetwork{ network };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpNetwork);
        }
      }

      template <typename T>
      Network<T>& Network<T>::operator=(const Network& network)
      {
        if (this != &network)
        {
          Network<T> tmpNetwork{ network };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpNetwork);
        }
        return *this;
      }

      template <typename T>
      NetworkTopology Network<T>::GetTopology() const
      {
        return Topology;
      }

      template <typename T>
      void Network<T>::SetTopology(const NetworkTopology& topology)
      {
        Network<T> tmpNetwork{ topology };
        // Beware, it is very intimate place for strong exception guarantee.
        std::swap(*this, tmpNetwork);
      }

      template <typename T>
      const Layer<T>& Network<T>::GetLayer(const size_t index) const
      {
        if (index >= Topology.GetLayerCount())
        {
          throw std::range_error("cnn::engine::perceptron::Network::GetLayer() const, index >= Topology.GetLayerCount().");
        }
        return Layers[index];
      }

      template <typename T>
      Layer<T> Network<T>::GetLayer(const size_t index)
      {
        if (index >= Topology.GetLayerCount())
        {
          throw std::range_error("cnn::engine::perceptron::Network::GetLayer(), index >= Topology.GetLayerCount().");
        }
        return Layers[index];
      }

      template <typename T>
      void Network<T>::GenerateOutput()
      {
        for (size_t l = 0; l < Topology.GetLayerCount(); ++l)
        {
          auto& currentLayer = Layers[l];
          if (l != 0)
          {
            const auto& previousLayer = Layers[l - 1];
            currentLayer.GetInput().FillFrom(previousLayer.GetOutput());
          }
          currentLayer.GenerateOutput();
        }
      }

      template <typename T>
      void Network<T>::Clear() noexcept
      {
        for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
        {
          Layers[i].Clear();
        }
      }

      template <typename T>
      void Network<T>::Reset() noexcept
      {
        Topology.Reset();
        Layers.reset(nullptr);
      }

      template <typename T>
      void Network<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::perceptron::Network::Save(), ostream.good() == false.");
        }

        Topology.Save(ostream);

        if (Topology.GetLayerCount() != 0)
        {
          for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
          {
            Layers[i].Save(ostream);
          }
        }

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::perceptron::Network::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Network<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::perceptron::Network::Load(), istream.good() == false.");
        }

        decltype(Topology) topology;
        decltype(Layers) layers;

        topology.Load(istream);
        CheckTopology(topology);

        if (topology.GetLayerCount() != 0)
        {
          layers = std::make_unique<Layer<T>[]>(topology.GetLayerCount());
          for (size_t i = 0; i < topology.GetLayerCount(); ++i)
          {
            layers[i].Load(istream);
            if (layers[i].GetTopology() != topology.GetLayerTopology(i))
            {
              throw std::logic_error("cnn::engine::perceptron::Network::Load(), layers[i].GetTopology() != topology.GetLayerTopology(i).");
            }
          }
        }

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::perceptron::Network::Load(), istream.good() == false.");
        }

        Topology = std::move(topology);
        Layers = std::move(layers);
      }

      template <typename T>
      void Network<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept
      {
        for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
        {
          Layers[i].FillWeights(valueGenerator);
        }
      }

      template <typename T>
      void Network<T>::Mutate(common::Mutagen<T>& mutagen) noexcept
      {
        for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
        {
          Layers[i].Mutate(mutagen);
        }
      }

      template <typename T>
      void Network<T>::CheckTopology(const NetworkTopology& topology) const
      {
        // Zero topology is allowed.
        if (topology.GetLayerCount() == 0)
        {
          return;
        }
        
        if (topology.GetLayerCount() > 1)
        {
          for (size_t i = 1; i < topology.GetLayerCount(); ++i)
          {
            const auto previousTopology = topology.GetLayerTopology(i - 1);
            const auto currentTopology = topology.GetLayerTopology(i);

            if (previousTopology.GetNeuronCount() != currentTopology.GetInputCount())
            {
              throw std::invalid_argument("cnn::engine::perceptron::Network::CheckTopology(), previousTopology.GetNeuronCount() != currentTopology.GetInputCount().");
            }
          }
        }
      }

    }
  }
}