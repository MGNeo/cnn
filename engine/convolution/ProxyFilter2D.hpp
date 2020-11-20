#pragma once

#include "Filter2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class ProxyFilter2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyFilter2D(Filter2D<T>& filter);

        // We delete copy-constructor too for saving of const-transitivity.
        // Now we can't build non-const object using const object.
        // What will you say about it, Elon Musk? ;-)
        ProxyFilter2D(const ProxyFilter2D& proxyFilter) = delete;

        ProxyFilter2D(ProxyFilter2D&& proxyFilter) = delete;

        ProxyFilter2D& operator=(const ProxyFilter2D& proxyFilter) = delete;

        ProxyFilter2D& operator=(ProxyFilter2D&& proxyFilter) = delete;

        Filter2DTopology GetTopology() const noexcept;

        // Exception guarantee: strong for this.
        const ProxyCore2D<T> GetCore(const size_t index) const;

        // Exception guarantee: strong for this.
        ProxyCore2D<T> GetCore(const size_t index);

        // It clears the state without changing of the topology.
        void Clear() noexcept;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) noexcept;

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
      Filter2DTopology ProxyFilter2D<T>::GetTopology() const noexcept
      {
        return Filter.GetTopology();
      }

      template <typename T>
      const ProxyCore2D<T> ProxyFilter2D<T>::GetCore(const size_t index) const
      {
        return Filter.GetCore(index);
      }

      template <typename T>
      ProxyCore2D<T> ProxyFilter2D<T>::GetCore(const size_t index)
      {
        return Filter.GetCore(index);
      }

      template <typename T>
      void ProxyFilter2D<T>::Clear() noexcept
      {
        Filter.Clear();
      }

      template <typename T>
      void ProxyFilter2D<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept
      {
        Filter.FillWeights(valueGenerator);
      }

      template <typename T>
      void ProxyFilter2D<T>::Mutate(common::Mutagen<T>& mutagen) noexcept
      {
        Filter.Mutate(mutagen);
      }
    }
  }
}