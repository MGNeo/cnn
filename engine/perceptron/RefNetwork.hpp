#pragma once

#include "Network.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      // RefNetwork is a wrapper which implements semantics of a safe reference on Network.
      // The safe reference allows to use methods of the target object, but doesn't allow methods which can change
      // the topology of the target object.
      // This is necessary to protect structure of a complex object which contains the target object as its part.
      template <typename T>
      class RefNetwork
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        RefNetwork(Network<T>& network) noexcept;

        RefNetwork(const RefNetwork& refNetwork) noexcept;

        RefNetwork(RefNetwork&& refNetwork) noexcept = delete;

        RefNetwork& operator=(const RefNetwork& refNetwork) noexcept = delete;

        RefNetwork& operator=(RefNetwork&& refNetwork) noexcept = delete;

        NetworkTopology GetTopology() const;

        // Exception guarantee: strong for the network.
        RefLayer<T>& GetLayer(const size_t index) const;

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
      RefNetwork<T>::RefNetwork(Network<T>& network) noexcept
        :
        Network_{ network }
      {
      }

      template <typename T>
      RefNetwork<T>::RefNetwork(const RefNetwork& refNetwork) noexcept
        :
        Network_{ refNetwork.Network_ }
      {
      }

      template <typename T>
      NetworkTopology RefNetwork<T>::GetTopology() const
      {
        return Network_.GetTopology();
      }

      template <typename T>
      RefLayer<T>& RefNetwork<T>::GetLayer(const size_t index) const
      {
        return Network_.GetLayer(index);
      }

      template <typename T>
      void RefNetwork<T>::GenerateOutput() const
      {
        Network_.GenerateOutput();
      }

      template <typename T>
      void RefNetwork<T>::Clear() const noexcept
      {
        Network_.Clear();
      }

      template <typename T>
      void RefNetwork<T>::Save(std::ostream& ostream) const
      {
        Network_.Save(ostream);
      }

      template <typename T>
      void RefNetwork<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Network_.FillWeights(valueGenerator);
      }

      template <typename T>
      void RefNetwork<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Network_.Mutate(mutagen);
      }
    }
  }
}