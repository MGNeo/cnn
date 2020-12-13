#pragma once

#include "Layer2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Layer2DProtectingReference
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Layer2DProtectingReference(Layer2D<T>& layer) noexcept;

        Layer2DProtectingReference(const Layer2DProtectingReference& layerReference) noexcept;

        Layer2DProtectingReference(Layer2DProtectingReference&& layerReference) noexcept = delete;

        Layer2DProtectingReference& operator=(const Layer2DProtectingReference& layerReference) noexcept = delete;

        Layer2DProtectingReference& operator=(Layer2DProtectingReference&& layerReference) noexcept = delete;

        const Layer2DTopology& GetTopology() const noexcept;

        // Exception guarantee: strong for the layer.
        const Map2D<T>& GetConstInput(const size_t index) const;

        // Exception guarantee: strong for the layer.
        Map2DProtectingReference<T> GetInput(const size_t index) const;

        // Exception guarantee: strong for the layer.
        const Filter2D<T>& GetConstFilter(const size_t index) const;

        // Exception guarantee: strong for the layer.
        Filter2DProtectingReference<T> GetFilter(const size_t index) const;

        // Exception guarantee: strong for the layer.
        const Map2D<T>& GetConstOutput(const size_t index) const;

        // Exception guarantee: strong for the layer.
        Map2DProtectingReference<T> GetOutput(const size_t index) const;

        // Exception guarantee: base for this.
        void GenerateOutput() const;

        // It clears the state without changing of the topology of the layer.
        void Clear() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) const noexcept;

      private:

        Layer2D<T>& Layer;

      };

      template <typename T>
      Layer2DProtectingReference<T>::Layer2DProtectingReference(Layer2D<T>& layer) noexcept
        :
        Layer{ layer }
      {
      }

      template <typename T>
      Layer2DProtectingReference<T>::Layer2DProtectingReference(const Layer2DProtectingReference& layerReference) noexcept
        :
        Layer{ layerReference.Layer }
      {
      }

      template <typename T>
      const Layer2DTopology& Layer2DProtectingReference<T>::GetTopology() const noexcept
      {
        return Layer.GetTopology();
      }

      template <typename T>
      const Map2D<T>& Layer2DProtectingReference<T>::GetConstInput(const size_t index) const
      {
        return Layer.GetInput(index);
      }

      template <typename T>
      Map2DProtectingReference<T> Layer2DProtectingReference<T>::GetInput(const size_t index) const
      {
        return Layer.GetInput(index);
      }

      template <typename T>
      const Filter2D<T>& Layer2DProtectingReference<T>::GetConstFilter(const size_t index) const
      {
        return Layer.GetFilter(index);
      }

      template <typename T>
      Filter2DProtectingReference<T> Layer2DProtectingReference<T>::GetFilter(const size_t index) const
      {
        return Layer.GetFilter(index);
      }

      template <typename T>
      const Map2D<T>& Layer2DProtectingReference<T>::GetConstOutput(const size_t index) const
      {
        return Layer.GetOutput(index);
      }

      template <typename T>
      Map2DProtectingReference<T> Layer2DProtectingReference<T>::GetOutput(const size_t index) const
      {
        return Layer.GetOutput(index);
      }

      template <typename T>
      void Layer2DProtectingReference<T>::GenerateOutput() const
      {
        Layer.GenerateOutput();
      }

      template <typename T>
      void Layer2DProtectingReference<T>::Clear() const noexcept
      {
        Layer.Clear();
      }

      template <typename T>
      void Layer2DProtectingReference<T>::Save(std::ostream& ostream) const
      {
        Layer.Save(ostream);
      }

      template <typename T>
      void Layer2DProtectingReference<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Layer.FillWeights(valueGenerator);
      }

      template <typename T>
      void Layer2DProtectingReference<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Layer.Mutate(mutagen);
      }
    }
  }
}