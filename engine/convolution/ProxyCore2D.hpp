#pragma once

#include "Core2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // ProxyCore2D is a protecting proxy, which protects a Core2D from changing of topology and other dangerous
      // operations, which can break consistency of complex object, which contains a Core2D as its part.
      template <typename T>
      class ProxyCore2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyCore2D(Core2D<T>& core) noexcept;

        ProxyCore2D(const ProxyCore2D& proxyCore) noexcept;

        ProxyCore2D(ProxyCore2D&& proxyCore) = delete;

        ProxyCore2D& operator=(const ProxyCore2D& proxyCore) = delete;

        ProxyCore2D& operator=(ProxyCore2D&& proxyCore) = delete;

        Size2D GetSize() const noexcept;

        T GetInput(const size_t x, const size_t y) const;

        // Exception guarantee: strong for the core.
        void SetInput(const size_t x, const size_t y, const T value) const;

        T GetWeight(const size_t x, const size_t y) const;

        // Exception guarantee: strong for the core.
        void SetWeight(const size_t x, const size_t y, const T value) const;

        void GenerateOutput() const;

        T GetOutput() const noexcept;

        // It clears the state without changing of the topology.
        void ClearInputs() const noexcept;

        // It clears the state without changing of the topology.
        void ClearWeights() const noexcept;

        // It clears the state without changing of the topology.
        void ClearOutput() const noexcept;

        // It clears the state without changing of the topology.
        void Clear() const noexcept;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) const noexcept;

      private:

        Core2D<T>& Core;

      };

      template <typename T>
      ProxyCore2D<T>::ProxyCore2D(Core2D<T>& core) noexcept
        :
        Core{ core }
      {
      }

      template <typename T>
      ProxyCore2D<T>::ProxyCore2D(const ProxyCore2D& proxyCore) noexcept
        :
        Core{ proxyCore.Core }
      {
      }

      template <typename T>
      Size2D ProxyCore2D<T>::GetSize() const noexcept
      {
        return Core.GetSize();
      }

      template <typename T>
      T ProxyCore2D<T>::GetInput(const size_t x, const size_t y) const
      {
        return Core.GetInput(x, y);
      }

      template <typename T>
      void ProxyCore2D<T>::SetInput(const size_t x, const size_t y, const T value) const
      {
        Core.SetInput(x, y, value);
      }

      template <typename T>
      T ProxyCore2D<T>::GetWeight(const size_t x, const size_t y) const
      {
        return Core.GetWeight(x, y);
      }

      template <typename T>
      void ProxyCore2D<T>::SetWeight(const size_t x, const size_t y, const T value) const
      {
        Core.SetWeight(x, y, value);
      }

      template <typename T>
      void ProxyCore2D<T>::GenerateOutput() const
      {
        Core.GenerateOutput();
      }

      template <typename T>
      T ProxyCore2D<T>::GetOutput() const noexcept
      {
        return Core.GetOutput();
      }

      template <typename T>
      void ProxyCore2D<T>::ClearInputs() const noexcept
      {
        Core.ClearInputs();
      }

      template <typename T>
      void ProxyCore2D<T>::ClearWeights() const noexcept
      {
        Core.ClearWeights();
      }

      template <typename T>
      void ProxyCore2D<T>::ClearOutput() const noexcept
      {
        Core.ClearOutput();
      }

      template <typename T>
      void ProxyCore2D<T>::Clear() const noexcept
      {
        Core.Clear();
      }

      template <typename T>
      void ProxyCore2D<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Core.FillWeights(valueGenerator);
      }

      template <typename T>
      void ProxyCore2D<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Core.Mutate(mutagen);
      }
    }
  }
}