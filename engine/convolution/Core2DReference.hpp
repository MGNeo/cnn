#pragma once

#include "Core2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      // Core2DReference is a type which implements semantics of smart reference to Core2D.
      // The smart reference proxies all methods of Core2D and doesn't allow to use methods, which change
      // the topology of the target core.
      // It allow to protect consistency of complex objects, which contain the target core as its part.
      template <typename T>
      class Core2DReference
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Core2DReference(Core2D<T>& core) noexcept;

        Core2DReference(const Core2DReference& coreReference) noexcept;

        Core2DReference(Core2DReference&& coreReference) noexcept = delete;
        
        Core2DReference& operator=(const Core2DReference& coreReference) noexcept = delete;

        Core2DReference& operator=(Core2DReference&& coreReference) noexcept = delete;

        const Size2D& GetSize() const noexcept;

        T GetInput(const size_t x, const size_t y) const;

        // Exception guarantee: strong for the core.
        void SetInput(const size_t x, const size_t y, const T value) const;

        T GetWeight(const size_t x, const size_t y) const;

        // Exception guarantee: strong for the core.
        void SetWeight(const size_t x, const size_t y, const T value) const;

        void GenerateOutput() const noexcept;

        T GetOutput() const noexcept;

        // It clears the state without changing of the topology of the core.
        void ClearInputs() const noexcept;

        // It clears the state without changing of the topology of the core.
        void ClearWeights() const noexcept;

        // It clears the state without changing of the topology of the core.
        void ClearOutput() const noexcept;

        // It clears the state without changing of the topology of the core.
        void Clear() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) const noexcept;

      private:

        Core2D<T>& Core;

      };

      template <typename T>
      Core2DReference<T>::Core2DReference(Core2D<T>& core) noexcept
        :
        Core{ core }
      {
      }

      template <typename T>
      Core2DReference<T>::Core2DReference(const Core2DReference& coreReference) noexcept
        :
        Core{ coreReference.Core }
      {
      }

      template <typename T>
      const Size2D& Core2DReference<T>::GetSize() const noexcept
      {
        return Core.GetSize();
      }

      template <typename T>
      T Core2DReference<T>::GetInput(const size_t x, const size_t y) const
      {
        return Core.GetInput(x, y);
      }

      template <typename T>
      void Core2DReference<T>::SetInput(const size_t x, const size_t y, const T value) const
      {
        Core.SetInput(x, y, value);
      }

      template <typename T>
      T Core2DReference<T>::GetWeight(const size_t x, const size_t y) const
      {
        return Core.GetWeight(x, y);
      }

      template <typename T>
      void Core2DReference<T>::SetWeight(const size_t x, const size_t y, const T value) const
      {
        Core.SetWeight(x, y, value);
      }

      template <typename T>
      void Core2DReference<T>::GenerateOutput() const noexcept
      {
        Core.GenerateOutput();
      }

      template <typename T>
      T Core2DReference<T>::GetOutput() const noexcept
      {
        return Core.GetOutput();
      }

      template <typename T>
      void Core2DReference<T>::ClearInputs() const noexcept
      {
        Core.ClearInputs();
      }

      template <typename T>
      void Core2DReference<T>::ClearWeights() const noexcept
      {
        Core.ClearWeights();
      }

      template <typename T>
      void Core2DReference<T>::ClearOutput() const noexcept
      {
        Core.ClearOutput();
      }

      template <typename T>
      void Core2DReference<T>::Clear() const noexcept
      {
        Core.Clear();
      }

      template <typename T>
      void Core2DReference<T>::Save(std::ostream& ostream) const
      {
        Core.Save(ostream);
      }

      template <typename T>
      void Core2DReference<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Core.FillWeights(valueGenerator);
      }

      template <typename T>
      void Core2DReference<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Core.Mutate(mutagen);
      }
    }
  }
}