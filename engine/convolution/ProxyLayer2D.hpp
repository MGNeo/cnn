#pragma once

#include "Layer2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class ProxyLayer2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyLayer2D(Layer2D<T>& layer);

        ProxyLayer2D(const ProxyLayer2D& proxyLayer) noexcept;

        ProxyLayer2D(ProxyLayer2D&& proxyLayer) = delete;

        ProxyLayer2D& operator=(const ProxyLayer2D& proxyLayer) = delete;

        ProxyLayer2D& operator=(ProxyLayer2D&& proxyLayer) = delete;

        Layer2DTopology GetTopology() const noexcept;

        ProxyMap2D<T> GetInput(const size_t index) const;

        ProxyFilter2D<T> GetFilter(const size_t index) const;

        ProxyMap2D<T> GetOutput(const size_t index) const;

        // Exception guarantee: base for this.
        void GenerateOputput();// TODO: noexcept?

        // It clears the state without changing of the topology.
        void Clear() noexcept;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) noexcept;

      private:

        Layer2D<T>& Layer;

      };

      template <typename T>
      ProxyLayer2D<T>::ProxyLayer2D(Layer2D<T>& layer)
        :
        Layer{ layer }
      {
      }

      template <typename T>
      ProxyLayer2D<T>::ProxyLayer2D(const ProxyLayer2D& proxyLayer) noexcept
        :
        Layer{ proxyLayer.Layer }
      {
      }

      template <typename T>
      Layer2DTopology ProxyLayer2D<T>::GetTopology() const noexcept
      {
        return Layer.GetTopology();
      }

      template <typename T>
      ProxyMap2D<T> ProxyLayer2D<T>::GetInput(const size_t index) const
      {
        return Layer.GetInput(index);
      }

      template <typename T>
      ProxyFilter2D<T> ProxyLayer2D<T>::GetFilter(const size_t index) const
      {
        return Layer.GetFilter(index);
      }

      template <typename T>
      ProxyMap2D<T> ProxyLayer2D<T>::GetOutput(const size_t index) const
      {
        return Layer.GetOutput(index);
      }

      template <typename T>
      void ProxyLayer2D<T>::GenerateOputput()
      {
        Layer.GenerateOputput();
      }

      template <typename T>
      void ProxyLayer2D<T>::Clear() noexcept
      {
        Layer.Clear();
      }

      template <typename T>
      void ProxyLayer2D<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept
      {
        Layer.FillWeights(valueGenerator);
      }

      template <typename T>
      void ProxyLayer2D<T>::Mutate(common::Mutagen<T>& mutagen) noexcept
      {
        Layer.Mutate(mutagen);
      }
    }
  }
}