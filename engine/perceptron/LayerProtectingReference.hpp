#pragma once

#include "Layer.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      // LayerProtectingReference is a type which implements semantics of protecting reference to Layer.
      // The smart reference proxies all methods of Layer and doesn't allow to use methods, which change
      // the topology of the target layer.
      // It allow to protect consistency of complex objects, which contain the target layer as its part.
      template <typename T>
      class LayerProtectingReference
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        LayerProtectingReference(Layer<T>& layer) noexcept;

        LayerProtectingReference(const LayerProtectingReference& layerReference) noexcept;

        LayerProtectingReference(LayerProtectingReference& layerReference) noexcept = delete;

        LayerProtectingReference& operator=(const LayerProtectingReference& layerReference) noexcept = delete;

        LayerProtectingReference& operator=(LayerProtectingReference&& layerReference) noexcept = delete;

        const LayerTopology& GetTopology() const noexcept;

        const common::Map<T>& GetConstInput() const noexcept;

        common::MapProtectingReference<T> GetInput() const noexcept;

        const common::Neuron<T>& GetConstNeuron(const size_t index) const;

        // Exception guarantee: strong for the layer.
        common::NeuronProtectingReference<T> GetNeuron(const size_t index) const;

        const common::Map<T>& GetConstOutput() const noexcept;

        common::MapProtectingReference<T> GetOutput() const noexcept;

        // Exception guarantee: base for the layer.
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

        Layer<T>& Layer_;

      };

      template <typename T>
      LayerProtectingReference<T>::LayerProtectingReference(Layer<T>& layer) noexcept
        :
        Layer_{ layer }
      {
      }

      template <typename T>
      LayerProtectingReference<T>::LayerProtectingReference(const LayerProtectingReference& layerReference) noexcept
        :
        Layer_{ layerReference.Layer_ }
      {
      }

      template <typename T>
      const LayerTopology& LayerProtectingReference<T>::GetTopology() const noexcept
      {
        return Layer_.GetTopology();
      }

      template <typename T>
      const common::Map<T>& LayerProtectingReference<T>::GetConstInput() const noexcept
      {
        return Layer_.GetInput();
      }

      template <typename T>
      common::MapProtectingReference<T> LayerProtectingReference<T>::GetInput() const noexcept
      {
        return Layer_.GetInput();
      }

      template <typename T>
      const common::Neuron<T>& LayerProtectingReference<T>::GetConstNeuron(const size_t index) const
      {
        return Layer_.GetConstNeuron(index);
      }

      template <typename T>
      common::NeuronProtectingReference<T> LayerProtectingReference<T>::GetNeuron(const size_t index) const
      {
        return Layer_.GetNeuron(index);
      }

      template <typename T>
      const common::Map<T>& LayerProtectingReference<T>::GetConstOutput() const noexcept
      {
        return Layer_.GetOutput();
      }

      template <typename T>
      common::MapProtectingReference<T> LayerProtectingReference<T>::GetOutput() const noexcept
      {
        return Layer_.GetOutput();
      }

      template <typename T>
      void LayerProtectingReference<T>::GenerateOutput() const
      {
        Layer_.GenerateOutput();
      }

      template <typename T>
      void LayerProtectingReference<T>::Clear() const noexcept
      {
        Layer_.Clear();
      }

      template <typename T>
      void LayerProtectingReference<T>::Save(std::ostream& ostream) const
      {
        Layer_.Save(ostream);
      }

      template <typename T>
      void LayerProtectingReference<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Layer_.FillWeights(valueGenerator);
      }

      template <typename T>
      void LayerProtectingReference<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Layer_.Mutate(mutagen);
      }
    }
  }
}
