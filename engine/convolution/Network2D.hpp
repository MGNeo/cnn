#pragma once

#include "Layer2D.hpp"
#include "Network2DTopology.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Network2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Network2D(const Network2DTopology& topology = {});

        Network2D(const Network2D& network);

        Network2D(Network2D&& network) noexcept = default;

        Network2D& operator=(const Network2D& network);

        Network2D& operator=(Network2D&& network) noexcept = default;

        Network2DTopology GetTopology() const;

        // Exception guarantee: base for this.
        void SetTopology(const Network2DTopology& topology);

        const Layer2D<T>& GetLayer(const size_t index) const;

        // Exception guarantee: base for this.
        Layer2D<T>& GetLayer(const size_t index);

        // Exception guarantee: base for this.
        void GenerateOputput();

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

        Network2DTopology Topology;
        std::unique_ptr<Layer2D<T>[]> Layers;

        void CheckTopology(const Network2DTopology& topology) const;

      };

      template <typename T>
      Network2D<T>::Network2D(const Network2DTopology& topology)
      {
        CheckTopology(topology);

        Topology = topology;

        if (Topology.GetLayerCount() != 0)
        {
          Layers = std::make_unique<Layer2D<T>[]>(Topology.GetLayerCount());
          for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
          {
            Layers[i].SetTopology(Topology.GetLayerTopology(i));
          }
        }
      }

      template <typename T>
      Network2D<T>::Network2D(const Network2D& network)
        :
        Topology{ network.Topology() }
      {
        if (Topology.GetLayerCount() != 0)
        {
          Layers = std::make_unique<Layer2D<T>[]>(Topology.GetLayerCount());
          for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
          {
            Layers[i] = network.Layers[i];
          }
        }
      }

      template <typename T>
      Network2D<T>& Network2D<T>::operator=(const Network2D& network)
      {
        if (this != &network)
        {
          Network2D<T> tmpNetwork{ network };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpNetwork);
        }
        return *this;
      }

      template <typename T>
      Network2DTopology Network2D<T>::GetTopology() const
      {
        return Topology;
      }

      template <typename T>
      void Network2D<T>::SetTopology(const Network2DTopology& topology)
      {
        Network2D<T> tmpNetwork{ topology };
        // Beware, it is very intimate place for strong exception guarantee.
        std::swap(*this, tmpNetwork);
      }

      template <typename T>
      const Layer2D<T>& Network2D<T>::GetLayer(const size_t index) const
      {
        if (index >= Topology.GetLayerCount())
        {
          throw std::range_error("cnn::engine::convolution::Network2D::GetLayer() const, index >= Topology.GetLayerCount().");
        }
        return Layers[index];
      }

      template <typename T>
      Layer2D<T>& Network2D<T>::GetLayer(const size_t index)
      {
        if (index >= Topology.GetLayerCount())
        {
          throw std::range_error("cnn::engine::convolution::Network2D::GetLayer(), index >= Topology.GetLayerCount().");
        }
        return Layers[index];
      }

      template <typename T>
      void Network2D<T>::GenerateOputput()
      {
        for (size_t l = 0; l < Topology.GetLayerCount(); ++l)
        {
          auto& currentLayer = Layers[l];
          if (l != 0)
          {
            const auto& topology = Topology.GetLayerTopology(l);
            const auto& previousLayer = Layers[l - 1];
            for (size_t i = 0; i < topology.GetInputCount(); ++i)
            {
              currentLayer.GetInput(i).FillFrom(previousLayer.GetOutput(i));
            }
          }
          currentLayer.GenerateOutput();
        }
      }

      template <typename T>
      void Network2D<T>::Clear() noexcept
      {
        for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
        {
          Layers[i].Clear();
        }
      }

      template <typename T>
      void Network2D<T>::Reset() noexcept
      {
        Topology.Reset();
        Layers.reset(nullptr);
      }

      template <typename T>
      void Network2D<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Network2D::Save(), ostream.good() == false.");
        }
        Topology.Save(ostream);
        for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
        {
          Layers[i].Save(ostream);
        }
        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Network2D::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Network2D<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Network2D::Load(), istream.good() == false.");
        }

        decltype(Topology) topology;
        decltype(Layers) layers;

        topology.Load(istream);
        CheckTopology(topology);

        if (topology.GetLayerCount() != 0)
        {
          layers = std::make_unique<Layer2D<T>[]>(topology.GetLayerCount());
          for (size_t i = 0; i < topology.GetLayerCount(); ++i)
          {
            layers[i].Load(istream);
            if (layers[i].GetTopology() != topology.GetLayerTopology(i))
            {
              throw std::logic_error("cnn::engine::convolution::Network2D::Load(), layers[i].GetTopology() != topology.GetLayerTopology(i).");
            }
          }
        }

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Network2D::Load(), istream.good() == false.");
        }

        Topology = std::move(topology);
        Layers = std::move(layers);
      }

      template <typename T>
      void Network2D<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept
      {
        for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
        {
          Layers[i].FillWeights(valueGenerator);
        }
      }

      template <typename T>
      void Network2D<T>::Mutate(common::Mutagen<T>& mutagen) noexcept
      {
        for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
        {
          Layers[i].Mutate(mutagen);
        }
      }

      template <typename T>
      void Network2D<T>::CheckTopology(const Network2DTopology& topology) const
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
            
            if (previousTopology.GetOutputSize() != currentTopology.GetInputSize())
            {
              throw std::invalid_argument("cnn::engine::convolution::Network2D::CheckTopology(), previousTopology.GetOutputSize() != currentTopology.GetInputSize().");
            }

            if (previousTopology.GetOutputCount() != currentTopology.GetInputCount())
            {
              throw std::invalid_argument("cnn::engine::convolution::Network2D::CheckTopology(), previousTopology.GetOutputCount() != currentTopology.GetInputCount().");
            }
          }
        }
      }
    }
  }
}