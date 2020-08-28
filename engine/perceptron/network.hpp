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

        void Process() override;

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
        if (index >= Layers.size())
        {
          throw std::range_error("cnn::engine::perceptron::Network::GetLayer() const, index >= Layers.size().");
        }
        return *(Layers[index]);
      }

      template <typename T>
      ILayer<T>& Network<T>::GetLayer(const size_t index)
      {
        if (index >= Layers.size())
        {
          throw std::range_error("cnn::engine::perceptron::Network::GetLayer(), index >= Layers.size().");
        }
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
    }
  }
}