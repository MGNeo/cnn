#pragma once

#include "Network2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // Network2DProtectingReference is a type which implements semantics of protecting reference to Network2D.
      // The protecting reference proxies all methods of Network2D and doesn't allow to use methods, which change
      // the topology of the target network.
      // It allow to protect consistency of complex objects, which contain the target network as its part.
      template <typename T>
      class Network2DProtectingReference
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Network2DProtectingReference(Network2D<T>& network) noexcept;

        Network2DProtectingReference(const Network2DProtectingReference& networkReference) noexcept;

        Network2DProtectingReference(Network2DProtectingReference&& networkReference) noexcept = delete;

        Network2DProtectingReference& operator=(const Network2DProtectingReference& networkReference) noexcept = delete;

        Network2DProtectingReference& operator=(Network2DProtectingReference&& networkReference) noexcept = delete;

        const Network2DTopology& GetTopology() const;

        const Layer2D<T>& GetConstLayer(const size_t index) const;

        // Exception guarantee: strong for the network.
        Layer2DProtectingReference<T> GetLayer(const size_t index) const;

        const Layer2D<T>& GetConstFirstLayer() const;

        // Exception guarantee: strong for the network.
        Layer2DProtectingReference<T> GetFirstLayer() const;

        const Layer2D<T>& GetConstLastLayer() const;

        // Exception guarantee: strong for the network.
        Layer2DProtectingReference<T> GetLastLayer() const;

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

        Network2D<T>& Network;

      };

      template <typename T>
      Network2DProtectingReference<T>::Network2DProtectingReference(Network2D<T>& network) noexcept
        :
        Network{ network }
      {
      }

      template <typename T>
      Network2DProtectingReference<T>::Network2DProtectingReference(const Network2DProtectingReference& networkReference) noexcept
        :
        Network{ networkReference.Network }
      {
      }

      template <typename T>
      const Network2DTopology& Network2DProtectingReference<T>::GetTopology() const
      {
        return Network.GetTopology();
      }

      template <typename T>
      const Layer2D<T>& Network2DProtectingReference<T>::GetConstLayer(const size_t index) const
      {
        return Network.GetLayer(index);
      }

      template <typename T>
      Layer2DProtectingReference<T> Network2DProtectingReference<T>::GetLayer(const size_t index) const
      {
        return Network.GetLayer(index);
      }

      template <typename T>
      const Layer2D<T>& Network2DProtectingReference<T>::GetConstFirstLayer() const
      {
        return Network.GetFirstLayer();
      }

      template <typename T>
      Layer2DProtectingReference<T> Network2DProtectingReference<T>::GetFirstLayer() const
      {
        return Network.GetFirstLayer();
      }

      template <typename T>
      const Layer2D<T>& Network2DProtectingReference<T>::GetConstLastLayer() const
      {
        return Network.GetLastLayer();
      }

      template <typename T>
      Layer2DProtectingReference<T> Network2DProtectingReference<T>::GetLastLayer() const
      {
        return Network.GetLastLayer();
      }

      template <typename T>
      void Network2DProtectingReference<T>::GenerateOutput() const
      {
        Network.GenerateOutput();
      }

      template <typename T>
      void Network2DProtectingReference<T>::Clear() const noexcept
      {
        Network.Clear();
      }

      template <typename T>
      void Network2DProtectingReference<T>::Save(std::ostream& ostream) const
      {
        Network.Save(ostream);
      }

      template <typename T>
      void Network2DProtectingReference<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Network.FillWeights(valueGenerator);
      }

      template <typename T>
      void Network2DProtectingReference<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Network.Mutate(mutagen);
      }
    }
  }
}