#pragma once

#include "Layer2D.hpp"
#include "ProxyLayer2D.hpp"
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

        void SetTopology(const Network2DTopology& topology);

        ProxyLayer2D<T> GetLayer(const size_t index);

        // Exception guarantee: base for this.
        void GenerateOputput();// TODO: noexcept?

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

        void CheckTopology(const Network2DTopology& topology);

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
        CheckTopology(topology);
        Network2D<T> tmpNetwork{ topology };
        // Beware, it is very intimate place for strong exception guarantee.
        std::swap(*this, tmpNetwork);
      }

      template <typename T>
      ProxyLayer2D<T> Network2D<T>::GetLayer(const size_t index)
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
        // ...
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
        // ...
      }

      template <typename T>
      void Network2D<T>::Load(std::istream& istream)
      {
        // ..
      }

      template <typename T>
      void Network2D<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept
      {
        for (size_t i = 0; i < Topology.GetLayerCount(); ++i)
        {
          Layers[i].FilLWeights(valueGenerator);
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
      void Network2D<T>::CheckTopology(const Network2DTopology& topology)
      {
        // Zero topology is allowed.
        // ...
      }
    }
  }
}