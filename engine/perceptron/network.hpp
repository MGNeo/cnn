#pragma once

#include "i_network.hpp"
#include "layer.hpp"

#include <vector>
#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      template <typename T>
      class Network : public INetwork<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<Network<T>>;

        Network(const size_t inputCountInFirstLayer,
                const size_t outputCountInFirstLayer);

        void PushBack(const size_t outputCountInNewLayer);

        size_t GetLayerCount() const override;

        const ILayer<T>& GetLayer(const size_t index) const override;
        ILayer<T>& GetLayer(const size_t index) override;

        const ILayer<T>& GetLastLayer() const override;
        ILayer<T>& GetLastLayer() override;;

        const ILayer<T>& GetFirstLayer() const override;
        ILayer<T>& GetFirstLayer() override;

        void Process() override;

        // The result must not be nullptr.
        typename INetwork<T>::Uptr Clone(const bool cloneState) const override;

        Network(const Network<T>& network, const bool cloneState);

        void FillWeights(common::IValueGenerator<T>& valueGenerator) override;

        void CrossFrom(const INetwork<T>& source1,
                       const INetwork<T>& source2,
                       common::IBinaryRandomGenerator& binaryRandomGenerator) override;

        void Mutate(common::IMutagen<T>& mutagen) override;

        void SetActivationFunctions(const common::IActivationFunction<T>& activationFunction) override;

      private:

        std::vector<typename ILayer<T>::Uptr> Layers;

      };

      template <typename T>
      Network<T>::Network(const size_t inputCountInFirstLayer,
                          const size_t outputCountInFirstLayer)
      {
        typename ILayer<T>::Uptr firstLayer = std::make_unique<Layer<T>>(inputCountInFirstLayer,
                                                                         outputCountInFirstLayer);
        Layers.push_back(std::move(firstLayer));
      }

      template <typename T>
      void Network<T>::PushBack(const size_t outputCountInNewLayer)
      {
        typename ILayer<T>::Uptr newLayer = std::make_unique<Layer<T>>(GetLastLayer().GetOutputSize(),
                                                                       outputCountInNewLayer);
        Layers.push_back(std::move(newLayer));
      }

      template <typename T>
      size_t Network<T>::GetLayerCount() const
      {
        return Layers.size();
      }

      template <typename T>
      const ILayer<T>& Network<T>::GetLayer(const size_t index) const
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= Layers.size())
        {
          throw std::range_error("cnn::engine::perceptron::Network::GetLayer() const, index >= Layers.size().");
        }
#endif
        return *(Layers[index]);
      }

      template <typename T>
      ILayer<T>& Network<T>::GetLayer(const size_t index)
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= Layers.size())
        {
          throw std::range_error("cnn::engine::perceptron::Network::GetLayer(), index >= Layers.size().");
        }
#endif
        return *(Layers[index]);
      }

      template <typename T>
      const ILayer<T>& Network<T>::GetLastLayer() const
      {
        if (Layers.size() == 0)
        {
          throw std::logic_error("cnn::engine::perceptron::Network::GetLastLayer() const, Layers.size() == 0.");
        }
        return *(Layers.back());
      }

      template <typename T>
      ILayer<T>& Network<T>::GetLastLayer()
      {
        if (Layers.size() == 0)
        {
          throw std::logic_error("cnn::engine::perceptron::Network::GetLastLayer(), Layers.size() == 0.");
        }
        return *(Layers.back());
      }

      template <typename T>
      const ILayer<T>& Network<T>::GetFirstLayer() const
      {
        return *(Layers[0]);
      }

      template <typename T>
      ILayer<T>& Network<T>::GetFirstLayer()
      {
        return *(Layers[0]);
      }

      template <typename T>
      void Network<T>::Process()
      {
        for (size_t l = 0; l < Layers.size(); ++l)
        {
          auto& layer = *(Layers[l]);
          if (l != 0)
          {
            const auto& prevLayer = *(Layers[l - 1]);
            for (size_t i = 0; i < layer.GetInputSize(); ++i)
            {
              auto& input = layer.GetInput();
              const auto& prevOutput = prevLayer.GetOutput();
              const T value = prevOutput.GetValue(i);
              input.SetValue(i, value);
            }
          }
          layer.Process();
        }
      }

      // The result must not be nullptr.
      template <typename T>
      typename INetwork<T>::Uptr Network<T>::Clone(const bool cloneState) const
      {
        return std::make_unique<Network<T>>(*this, cloneState);
      }

      template <typename T>
      Network<T>::Network(const Network<T>& network, const bool cloneState)
      {
        Layers.resize(network.GetLayerCount());
        for (size_t l = 0; l < network.GetLayerCount(); ++l)
        {
          Layers[l] = network.GetLayer(l).Clone(cloneState);
        }
      }

      template <typename T>
      void Network<T>::FillWeights(common::IValueGenerator<T>& valueGenerator)
      {
        for (auto& layer : Layers)
        {
          layer->FillWeights(valueGenerator);
        }
      }

      template <typename T>
      void Network<T>::CrossFrom(const INetwork<T>& source1,
                                 const INetwork<T>& source2,
                                 common::IBinaryRandomGenerator& binaryRandomGenerator)
      {
        if (GetLayerCount() != source1.GetLayerCount())
        {
          throw std::invalid_argument("cnn::engine::perceptron::Network::CrossFrom(), GetLayerCount() != source1.GetLayerCount().");
        }
        if (GetLayerCount() != source2.GetLayerCount())
        {
          throw std::invalid_argument("cnn::engine::perceptron::Network::CrossFrom(), GetLayerCount() != source2.GetLayerCount().");
        }
        for (size_t l = 0; l < GetLayerCount(); ++l)
        {
          Layers[l]->CrossFrom(source1.GetLayer(l), source2.GetLayer(l), binaryRandomGenerator);
        }
      }

      template <typename T>
      void Network<T>::Mutate(common::IMutagen<T>& mutagen)
      {
        for (auto& layer : Layers)
        {
          layer->Mutate(mutagen);
        }
      }

      template <typename T>
      void Network<T>::SetActivationFunctions(const common::IActivationFunction<T>& activationFunction)
      {
        for (auto& layer : Layers)
        {
          layer->SetActivationFunctions(activationFunction);
        }
      }

    }
  }
}