#pragma once

#include "Network.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      // NetworkProtectingReference is a type which implements semantics of protecting reference to Network.
      // The protecting reference proxies all methods of Network and doesn't allow to use methods, which change
      // the topology of the target network.
      // It allow to protect consistency of complex objects, which contain the target network as its part.
      template <typename T>
      class NetworkProtectingReference
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        NetworkProtectingReference(Network<T>& network) noexcept;

        NetworkProtectingReference(const NetworkProtectingReference& networkReference) noexcept;

        NetworkProtectingReference(NetworkProtectingReference&& networkReference) noexcept = delete;

        NetworkProtectingReference& operator=(const NetworkProtectingReference& networkReference) noexcept = delete;

        NetworkProtectingReference& operator=(NetworkProtectingReference&& networkReference) noexcept = delete;

        const NetworkTopology& GetTopology() const;

        const Layer<T>& GetConstLayer(const size_t index) const;

        // Exception guarantee: strong for the network.
        LayerProtectingReference<T> GetLayer(const size_t index) const;

        const Layer<T>& GetConstFirstLayer() const;

        // Exception guarantee: strong for the network.
        LayerProtectingReference<T> GetFirstLayer() const;

        // Exception guarantee: base for the network.
        void GenerateOutput() const;

        // It clears the state without changing of the topology of the network.
        void Clear() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) const noexcept;

      private:

        Network<T>& Network_;

      };

      template <typename T>
      NetworkProtectingReference<T>::NetworkProtectingReference(Network<T>& network) noexcept
        :
        Network_{ network }
      {
      }

      template <typename T>
      NetworkProtectingReference<T>::NetworkProtectingReference(const NetworkProtectingReference& networkReference) noexcept
        :
        Network{ networkReference.Network_ }
      {
      }

      template <typename T>
      const NetworkTopology& NetworkProtectingReference<T>::GetTopology() const
      {
        return Network_.GetTopology();
      }

      template <typename T>
      const Layer<T>& NetworkProtectingReference<T>::GetConstLayer(const size_t index) const
      {
        return Network_.GetLayer(index);
      }

      template <typename T>
      LayerProtectingReference<T> NetworkProtectingReference<T>::GetLayer(const size_t index) const
      {
        return Network_.GetLayer(index);
      }

      template <typename T>
      const Layer<T>& NetworkProtectingReference<T>::GetConstFirstLayer() const
      {
        return Network_.GetFirstLayer();
      }

      template <typename T>
      LayerProtectingReference<T> NetworkProtectingReference<T>::GetFirstLayer() const
      {
        return Network_.GetFirstLayer();
      }

      template <typename T>
      void NetworkProtectingReference<T>::GenerateOutput() const
      {
        Network_.GenerateOutput();
      }

      template <typename T>
      void NetworkProtectingReference<T>::Clear() const noexcept
      {
        Network_.Clear();
      }

      template <typename T>
      void NetworkProtectingReference<T>::Save(std::ostream& ostream) const
      {
        Network_.Save(ostream);
      }

      template <typename T>
      void NetworkProtectingReference<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Network_.FillWeights(valueGenerator);
      }

      template <typename T>
      void NetworkProtectingReference<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Network_.Mutate(mutagen);
      }
    }
  }
}