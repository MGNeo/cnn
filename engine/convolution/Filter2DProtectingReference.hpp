#pragma once

#include "Filter2D.hpp"
#include "Core2DProtectingReference.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // Filter2DProtectingReference is a type which implements semantics of protecting reference to Filter2D.
      // The smart reference proxies all methods of Filter2D and doesn't allow to use methods, which change
      // the topology of the target filter.
      // It allow to protect consistency of complex objects, which contain the target filter as its part.
      template <typename T>
      class Filter2DProtectingReference
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Filter2DProtectingReference(Filter2D<T>& filter) noexcept;

        Filter2DProtectingReference(const Filter2DProtectingReference& filterReference) noexcept;

        Filter2DProtectingReference(Filter2DProtectingReference&& filterReference) noexcept = delete;

        Filter2DProtectingReference& operator=(const Filter2DProtectingReference& filterReference) noexcept = delete;

        Filter2DProtectingReference& operator=(Filter2DProtectingReference&& filterReference) noexcept = delete;

        const Filter2DTopology& GetTopology() const noexcept;

        const Core2D<T>& GetConstCore(const size_t index) const;

        // Exception guarantee: strong for the filter.
        Core2DProtectingReference<T> GetCore(const size_t index) const;

        // It clears the state without changing of the topology of the filter.
        void Clear() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) const noexcept;

      private:

        Filter2D<T>& Filter;
        
      };

      template <typename T>
      Filter2DProtectingReference<T>::Filter2DProtectingReference(Filter2D<T>& filter) noexcept
        :
        Filter{ filter }
      {
      }

      template <typename T>
      Filter2DProtectingReference<T>::Filter2DProtectingReference(const Filter2DProtectingReference& filterReference) noexcept
        :
        Filter{ filterReference.Filter }
      {
      }

      template <typename T>
      const Filter2DTopology& Filter2DProtectingReference<T>::GetTopology() const noexcept
      {
        return Filter.GetTopology();
      }

      template <typename T>
      const Core2D<T>& Filter2DProtectingReference<T>::GetConstCore(const size_t index) const
      {
        return Filter.GetCore(index);
      }

      template <typename T>
      Core2DProtectingReference<T> Filter2DProtectingReference<T>::GetCore(const size_t index) const
      {
        return Filter.GetCore(index);
      }

      template <typename T>
      void Filter2DProtectingReference<T>::Clear() const noexcept
      {
        Filter.Clear();
      }

      template <typename T>
      void Filter2DProtectingReference<T>::Save(std::ostream& ostream) const
      {
        Filter.Save(ostream);
      }

      template <typename T>
      void Filter2DProtectingReference<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Filter.FillWeights(valueGenerator);
      }

      template <typename T>
      void Filter2DProtectingReference<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Filter.Mutate(mutagen);
      }
    }
  }
}