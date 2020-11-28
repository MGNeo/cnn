#pragma once

#include "Layer.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      // ProxyLayer is a protecting proxy, which protects a Layer from changing of topology and other dangerous
      // operations, which can break consistency of complex object, which contains a Layer as its part.
      template <typename T>
      class ProxyLayer
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        ProxyLayer(Layer<T>& layer) noexcept;

        ProxyLayer(const ProxyLayer& proxyLayer) noexcept;

        ProxyLayer(ProxyLayer&& proxyLayer) = delete;

        ProxyLayer& operator=(const ProxyLayer& proxyLayer) = delete;

        ProxyLayer& operator=(ProxyLayer&& proxyLayer) = delete;

        LayerTopology GetTopology() const noexcept;

        common::ProxyMap<T> GetInput() const;

        common::ProxyNeuron<T> GetNeuron(const size_t index) const;

        common::ProxyMap<T> GetOutput() const;

        // Exception guarantee: base for this.
        void GenerateOutput() const;

        // It clears the state without changing of the topology.
        void Clear() const noexcept;

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) const noexcept;

      private:

        Layer<T>& Layer_;

      };

      template <typename T>
      ProxyLayer<T>::ProxyLayer(Layer<T>& layer) noexcept
        :
        Layer_{ layer }
      {
      }

      template <typename T>
      ProxyLayer<T>::ProxyLayer(const ProxyLayer& proxyLayer) noexcept
        :
        Layer_{ proxyLayer.Layer }
      {
      }

      template <typename T>
      LayerTopology ProxyLayer<T>::GetTopology() const noexcept
      {
        return Layer_.GetTopology();
      }

      template <typename T>
      common::ProxyMap<T> ProxyLayer<T>::GetInput() const
      {
        return Layer_.GetInput();
      }

      template <typename T>
      common::ProxyNeuron<T> ProxyLayer<T>::GetNeuron(const size_t index) const
      {
        return Layer_.GetNeuron(index);
      }

      template <typename T>
      common::ProxyMap<T> ProxyLayer<T>::GetOutput() const
      {
        return Layer_.GetOutput();
      }

      template <typename T>
      void ProxyLayer<T>::GenerateOutput() const
      {
        Layer_.GenerateOutput();
      }

      template <typename T>
      void ProxyLayer<T>::Clear() const noexcept
      {
        Layer_.Clear();
      }

      template <typename T>
      void ProxyLayer<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) const noexcept
      {
        Layer_.FillWeights(valueGenerator);
      }

      template <typename T>
      void ProxyLayer<T>::Mutate(common::Mutagen<T>& mutagen) const noexcept
      {
        Layer_.Mutate(mutagen);
      }
    }
  }
}