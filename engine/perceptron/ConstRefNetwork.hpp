#pragma once

#include "Network.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      // ConstRefNetwork is a wrapper which implements semantics of a safe const reference on Network.
      // The safe reference allows to use methods of the target object, but doesn't allow methods which can change
      // the topology of the target object.
      // This is necessary to protect structure of a complex object which contains the target object as its part.
      template <typename T>
      class ConstRefNetwork
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ConstRefNetwork(const Network<T>& network) noexcept;

        ConstRefNetwork(const ConstRefNetwork& constRefNetwork) noexcept;

        ConstRefNetwork(ConstRefNetwork&& constRefNetwork) noexcept = delete;

        ConstRefNetwork& operator=(const ConstRefNetwork& constRefNetwork) noexcept = delete;

        ConstRefNetwork& operator=(ConstRefNetwork&& constRefNetwork) noexcept = delete;

        NetworkTopology GetTopology() const;

        // Exception guarantee: strong for the network.
        ConstRefLayer<T>& GetLayer(const size_t index) const;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

      private:

        const Network<T>& Network_;

      };

      template <typename T>
      ConstRefNetwork<T>::ConstRefNetwork(const Network<T>& network) noexcept
        :
        Network_{ network }
      {
      }

      template <typename T>
      ConstRefNetwork<T>::ConstRefNetwork(const ConstRefNetwork& constRefNetwork) noexcept
        :
        Network_{ constRefNetwork.Network_ }
      {
      }

      template <typename T>
      NetworkTopology ConstRefNetwork<T>::GetTopology() const
      {
        return Network_.GetTopology();
      }

      template <typename T>
      ConstRefLayer<T>& ConstRefNetwork<T>::GetLayer(const size_t index) const
      {
        return Network_.GetLayer(index);
      }

      template <typename T>
      void ConstRefNetwork<T>::Save(std::ostream& ostream) const
      {
        Network_.Save(ostream);
      }
    }
  }
}