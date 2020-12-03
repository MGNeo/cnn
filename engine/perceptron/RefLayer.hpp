#pragma once

#include "Layer.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      // RefLayer is a wrapper which implements semantics of a safe reference on Layer.
      // The safe reference allows to use methods of the target object, but doesn't allow methods which can change
      // the topology of the target object.
      // This is necessary to protect structure of a complex object which contains the target object as its part.
      template <typename T>
      class RefLayer
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        RefLayer(Layer<T>& layer) noexcept;

        RefLayer(const RefLayer<T>& refLayer) noexcept;

        RefLayer(RefLayer<T>&& refLayer) noexcept = delete;

        RefLayer& operator=(const RefLayer& refLayer) noexcept = delete;

        RefLayer& operator=(RefLayer&& refLayaer) noexcept = delete;

        LayerTopology GetTopology() const noexcept;

        common::RefMap<T>& GetInput() const noexcept;

        // Exception guarantee: strong for the layer.
        common::RefNeuron<T>& GetNeuron(const size_t index) const;

        common::RefMap<T>& GetOutput() const noexcept;

        // Exception guarantee: base for the layer.
        void GenerateOutput() const;

        // It clears the state without changing of the topology of the layer.
        void Clear() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) const noexcept;

      private:

        Layer<T>& Layer_;

      };

      template <typename T>
      RefLayer<T>::RefLayer(Layer<T>& layer) noexcept
        :
        Layer_{ layer }
      {
      }

      template <typename T>
      RefLayer<T>::RefLayer(const RefLayer<T>& refLayer) noexcept
        :
        Layer_{ refLayer.Layer_ }
      {
      }

      template <typename T>
      LayerTopology RefLayer<T>::GetTopology() const noexcept
      {
        return Layer_.GetTopology();
      }

      template <typename T>
      common::RefMap<T>& RefLayer<T>::GetInput() const noexcept
      {
        return Layer_.GetInput();
      }

      template <typename T>
      common::RefNeuron<T>& RefLayer<T>::GetNeuron(const size_t index) const
      {
        return Layer_.GetNeuron(index);
      }

      template <typename T>
      common::RefMap<T>& RefLayer<T>::GetOutput() const noexcept
      {
        return Layer_.GetOutput();
      }

      template <typename T>
      void RefLayer<T>::GenerateOutput() const
      {
        Layer_.GenerateOutput();
      }

      template <typename T>
      void RefLayer<T>::Clear() const noexcept
      {
        Layer_.Clear();
      }

      template <typename T>
      void RefLayer<T>::Save(std::ostream& ostream) const
      {
        Layer_.Save(ostream);
      }

      template <typename T>
      void RefLayer<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept
      {
        Layer_.FillWeights(valueGenerator);
      }

      template <typename T>
      void RefLayer<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Layer_.Mutate(mutagen);
      }
    }
  }
}