#pragma once

#include "Filter2D.hpp"
#include "Core2DReference.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // Filter2DReference is a type which implements semantics of smart reference to Neuron.
      // The smart reference proxies all methods of Filter2D and doesn't allow to use methods, which change
      // the topology of the target filter.
      // It allow to protect consistency of complex objects, which contain the target filter as its part.
      template <typename T>
      class Filter2DReference
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Filter2DReference(Filter2D<T>& filter) noexcept;

        Filter2DReference(const Filter2DReference& filterReference) noexcept;

        Filter2DReference(Filter2DReference&& filterReference) noexcept = delete;

        Filter2DReference& operator=(const Filter2DReference& filterReference) noexcept = delete;

        Filter2DReference& operator=(Filter2DReference&& filterReference) noexcept = delete;

        const Filter2DTopology& GetTopology() const noexcept;

        const Core2D<T>& GetConstCore(const size_t index) const;

        // Exception guarantee: strong for the filter.
        Core2DReference<T>& GetCore(const size_t index) const;

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
      Filter2DReference<T>::Filter2DReference(Filter2D<T>& filter) noexcept
        :
        Filter{ filter }
      {
      }

      template <typename T>
      Filter2DReference<T>::Filter2DReference(const Filter2DReference& filterReference) noexcept
        :
        Filter{ filterReference.Filter }
      {
      }

      template <typename T>
      const Filter2DTopology& Filter2DReference<T>::GetTopology() const noexcept
      {
        return Filter.GetTopology();
      }

      template <typename T>
      const Core2D<T>& Filter2DReference<T>::GetConstCore(const size_t index) const
      {
        return Filter.GetCore(index);
      }

      template <typename T>
      Core2DReference<T>& Filter2DReference<T>::GetCore(const size_t index) const
      {
        return Filter.GetCore(index);
      }

      template <typename T>
      void Filter2DReference<T>::Clear() const noexcept
      {
        Filter.Clear();
      }

      template <typename T>
      void Filter2DReference<T>::Save(std::ostream& ostream) const
      {
        Filter.Save(ostream);
      }

      template <typename T>
      void Filter2DReference<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Filter.FillWeights(valueGenerator);
      }

      template <typename T>
      void Filter2DReference<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Filter.Mutate(mutagen);
      }
    }
  }
}