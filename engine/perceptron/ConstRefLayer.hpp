#pragma once

#include "Layer.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      // ConstRefLayer is a wrapper which implements semantics of a safe const reference on Layer.
      // The safe const reference allows to use methods of the target object, but doesn't allow methods which can change
      // the topology of the target object.
      // This is necessary to protect structure of a complex object which contains the target object as its part.
      template <typename T>
      class ConstRefLayer
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ConstRefLayer(const Layer<T>& layer) noexcept;

        ConstRefLayer(const ConstRefLayer<T>& constRefLayer) noexcept;

        ConstRefLayer(ConstRefLayer<T>&& constRefLayer) noexcept = delete;

        ConstRefLayer& operator=(const ConstRefLayer& constRefLayer) noexcept = delete;

        ConstRefLayer& operator=(ConstRefLayer&& constRefLayer) noexcept = delete;

        LayerTopology GetTopology() const noexcept;

        common::ConstRefMap<T>& GetInput() const noexcept;

        // Exception guarantee: strong for the layer.
        common::ConstRefNeuron<T>& GetNeuron(const size_t index) const;

        common::ConstRefMap<T>& GetOutput() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

      private:

        const Layer<T>& Layer_;

      };

      template <typename T>
      ConstRefLayer<T>::ConstRefLayer(const Layer<T>& layer) noexcept
        :
        Layer_{ layer }
      {
      }

      template <typename T>
      ConstRefLayer<T>::ConstRefLayer(const ConstRefLayer<T>& constRefLayer) noexcept
        :
        Layer_{ constRefLayer.Layer_ }
      {
      }

      template <typename T>
      LayerTopology ConstRefLayer<T>::GetTopology() const noexcept
      {
        return Layer_.GetTopology();
      }

      template <typename T>
      common::ConstRefMap<T>& ConstRefLayer<T>::GetInput() const noexcept
      {
        return Layer_.GetInput();
      }

      template <typename T>
      common::ConstRefNeuron<T>& ConstRefLayer<T>::GetNeuron(const size_t index) const
      {
        return Layer_.GetNeuron(index);
      }

      template <typename T>
      common::ConstRefMap<T>& ConstRefLayer<T>::GetOutput() const noexcept
      {
        return Layer_.GetOutput();
      }

      template <typename T>
      void ConstRefLayer<T>::Save(std::ostream& ostream) const
      {
        Layer_.Save(ostream);
      }
    }
  }
}