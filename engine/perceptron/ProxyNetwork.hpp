#pragma once

#include "Network.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      // ProxyNetwork is a protecting proxy, which protects a Network from changing of topology and other dangerous
      // operations, which can break consistency of complex object, which contains a Network as its part.
      template <typename T>
      class ProxyNetwork
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyNetwork(Network<T>& network) noexcept;

        ProxyNetwork(const ProxyNetwork& proxyNetwork) noexcept;

        ProxyNetwork(ProxyNetwork&& proxyNetwork) noexcept = delete;

        ProxyNetwork& operator=(const ProxyNetwork& proxyNetwork) noexcept = delete;

        ProxyNetwork& operator=(ProxyNetwork&& proxyNetwork) noexcept = delete;

        NetworkTopology GetTopology() const;

        ProxyLayer<T> GetLayer(const size_t index) const;

        // Exception guarantee: base for the network.
        void GenerateOutput() const;

        // It clears the state without changing of the topology.
        void Clear() const noexcept;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) const noexcept;

      private:

        Network<T>& Network_;

      };

      template <typename T>
      ProxyNetwork<T>::ProxyNetwork(Network<T>& network) noexcept
        :
        Network_{ network }
      {

      }

      template <typename T>
      ProxyNetwork<T>::ProxyNetwork(const ProxyNetwork& proxyNetwork) noexcept
        :
        Network_{ proxyNetwork.Network_ }
      {
      }

      template <typename T>
      NetworkTopology ProxyNetwork<T>::GetTopology() const
      {
        return Network_.GetTopology();
      }

      template <typename T>
      ProxyLayer<T> ProxyNetwork<T>::GetLayer(const size_t index) const
      {
        return Network_.GetLayer(index);
      }
      
      template <typename T>
      void ProxyNetwork<T>::GenerateOutput() const
      {
        Network_.GenerateOutput();
      }

      template <typename T>
      void ProxyNetwork<T>::Clear() const noexcept
      {
        Network_.Clear();
      }

      template <typename T>
      void ProxyNetwork<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Network_.FillWeights(valueGenerator);
      }

      template <typename T>
      void ProxyNetwork<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Network_.Mutate(mutagen);
      }
    }
  }
}