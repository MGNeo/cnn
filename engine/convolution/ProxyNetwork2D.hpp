#pragma once

#include "Network2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // ProxyNetwork2D is a protecting proxy, which protects a Network2D from changing of topology and other dangerous
      // operations, which can break consistency of complex object, which contains a Network2D as its part.
      template <typename T>
      class ProxyNetwork2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyNetwork2D(Network2D<T>& network) noexcept;

        ProxyNetwork2D(const ProxyNetwork2D& proxyNetwork) noexcept;

        ProxyNetwork2D(ProxyNetwork2D&& proxyNetwork) = delete;

        ProxyNetwork2D& operator=(const ProxyNetwork2D& proxyNetwork) = delete;

        ProxyNetwork2D& operator=(ProxyNetwork2D&& proxyNetwork) = delete;

        Network2DTopology GetTopology() const;

        // Exception guarantee: base for the network.
        void SetTopology(const Network2DTopology& topology) const;

        // Exception guarantee: base for the network.
        ProxyLayer2D<T> GetLayer(const size_t index) const;

        // Exception guarantee: base for the network.
        void GenerateOputput() const;

        // It clears the state without changing of the topology.
        void Clear() const noexcept;

        // It resets the state to zero including the topology.
        void Reset() const noexcept;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) const noexcept;

      private:

        Network2D<T>& Network;

      };

      template <typename T>
      ProxyNetwork2D<T>::ProxyNetwork2D(Network2D<T>& network) noexcept
        :
        Network{ network }
      {
      }

      template <typename T>
      ProxyNetwork2D<T>::ProxyNetwork2D(const ProxyNetwork2D& proxyNetwork) noexcept
        :
        Network{ proxyNetwork.Network }
      {
      }

      template <typename T>
      Network2DTopology ProxyNetwork2D<T>::GetTopology() const
      {
        return Network.GetTopology();
      }

      template <typename T>
      void ProxyNetwork2D<T>::SetTopology(const Network2DTopology& topology) const
      {
        Network.SetTopology(topology);
      }

      template <typename T>
      ProxyLayer2D<T> ProxyNetwork2D<T>::GetLayer(const size_t index) const
      {
        return Network.GetLayer(index);
      }

      template <typename T>
      void ProxyNetwork2D<T>::GenerateOputput() const
      {
        Network.GenerateOputput();
      }

      template <typename T>
      void ProxyNetwork2D<T>::Clear() const noexcept
      {
        Network.Clear();
      }

      template <typename T>
      void ProxyNetwork2D<T>::Reset() const noexcept
      {
        Network.Reset();
      }

      template <typename T>
      void ProxyNetwork2D<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Network.FillWeights(valueGenerator);
      }

      template <typename T>
      void ProxyNetwork2D<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Network.Mutate(mutagen);
      }
    }
  }
}