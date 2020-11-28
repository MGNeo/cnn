#pragma once

#include "Filter2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // ProxyFilter2D is a protecting proxy, which protects a Filter2D from changing of topology and other dangerous
      // operations, which can break consistency of complex object, which contains a Filter2D as its part.
      template <typename T>
      class ProxyFilter2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyFilter2D(Filter2D<T>& filter);

        ProxyFilter2D(const ProxyFilter2D& proxyFilter) noexcept;

        ProxyFilter2D(ProxyFilter2D&& proxyFilter) = delete;

        ProxyFilter2D& operator=(const ProxyFilter2D& proxyFilter) = delete;

        ProxyFilter2D& operator=(ProxyFilter2D&& proxyFilter) = delete;

        Filter2DTopology GetTopology() const noexcept;

        // Exception guarantee: strong for the filter.
        ProxyCore2D<T> GetCore(const size_t index) const;

        // It clears the state without changing of the topology.
        void Clear() const noexcept;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) const noexcept;

      private:

        Filter2D<T>& Filter;

      };

      template <typename T>
      ProxyFilter2D<T>::ProxyFilter2D(Filter2D<T>& filter)
        :
        Filter{ filter }
      {
      }

      template <typename T>
      ProxyFilter2D<T>::ProxyFilter2D(const ProxyFilter2D& proxyFilter) noexcept
        :
        Filter{ proxyFilter.Filter }
      {
      }

      template <typename T>
      Filter2DTopology ProxyFilter2D<T>::GetTopology() const noexcept
      {
        return Filter.GetTopology();
      }

      template <typename T>
      ProxyCore2D<T> ProxyFilter2D<T>::GetCore(const size_t index) const
      {
        return Filter.GetCore(index);
      }

      template <typename T>
      void ProxyFilter2D<T>::Clear() const noexcept
      {
        Filter.Clear();
      }

      template <typename T>
      void ProxyFilter2D<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Filter.FillWeights(valueGenerator);
      }

      template <typename T>
      void ProxyFilter2D<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Filter.Mutate(mutagen);
      }
    }
  }
}