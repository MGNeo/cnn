#pragma once

#include "i_network_2d.hpp"
#include "convolution_layer_2d.hpp"
#include "pooling_layer_2d.hpp"

#include <vector>
#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Network2D : public INetwork2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<Network2D<T>>;

        Network2D(const size_t inputWidthInFirstLayer,
                  const size_t inputHeightInFirstLayer,
                  const size_t inputCountInFirstLayer,

                  const size_t filterWidthInFirstLayer,
                  const size_t filterHeightInFirstLayer,
                  const size_t filterCountInFirstLayer);

        // TODO: Perhaps, there are situations where it will be useful to have
        // pooling layer in the beginning, so it needs to write special constructor for it.
        // ...
        
        void PushBack(const size_t stepSizeInNewLayer);

        void PushBack(const size_t filterWidthInNewLayer,
                      const size_t filterHeightInNewLayer,
                      const size_t filterCountInNewLayer);

        size_t GetLayerCount() const override;

        const ILayer2D<T>& GetLayer(const size_t index) const override;
        ILayer2D<T>& GetLayer(const size_t index) override;

        const ILayer2D<T>& GetLastLayer() const override;
        ILayer2D<T>& GetLastLayer() override;

        const ILayer2D<T>& GetFirstLayer() const override;
        ILayer2D<T>& GetFirstLayer() override;

        void Process() override;

      private:

        std::vector<typename ILayer2D<T>::Uptr> Layers;

      };

      template <typename T>
      Network2D<T>::Network2D(const size_t inputWidthInFirstLayer,
                              const size_t inputHeightInFirstLayer,
                              const size_t inputCountInFirstLayer,

                              const size_t filterWidthInFirstLayer,
                              const size_t filterHeightInFirstLayer,
                              const size_t filterCountInFirstLayer)
      {
        typename ILayer2D<T>::Uptr firstLayer = std::make_unique<ConvolutionLayer2D<T>>(inputWidthInFirstLayer,
                                                                                        inputHeightInFirstLayer,
                                                                                        inputCountInFirstLayer,

                                                                                        filterWidthInFirstLayer,
                                                                                        filterHeightInFirstLayer,
                                                                                        filterCountInFirstLayer);
        Layers.push_back(std::move(firstLayer));
      }

      template <typename T>
      void Network2D<T>::PushBack(const size_t stepSizeInNewLayer)
      {
        const auto& prevLayer = GetLastLayer();
        typename ILayer2D<T>::Uptr layer = std::make_unique<PoolingLayer2D<T>>(prevLayer,
                                                                               stepSizeInNewLayer);
        Layers.push_back(std::move(layer));
      }

      template <typename T>
      void Network2D<T>::PushBack(const size_t filterWidthInNewLayer,
                                  const size_t filterHeightInNewLayer,
                                  const size_t filterCountInNewLayer)
      {
        const auto& prevLayer = GetLastLayer();
        typename ILayer2D<T>::Uptr layer = std::make_unique<ConvolutionLayer2D<T>>(prevLayer,
                                                                                   filterWidthInNewLayer,
                                                                                   filterHeightInNewLayer,
                                                                                   filterCountInNewLayer);
        Layers.push_back(std::move(layer));
      }

      template <typename T>
      size_t Network2D<T>::GetLayerCount() const
      {
        return Layers.size();
      }

      template <typename T>
      const ILayer2D<T>& Network2D<T>::GetLayer(const size_t index) const
      {
        if (index >= Layers.size())
        {
          throw std::range_error("cnn::engine::convolution::Network2D::GetLayer() const, index >= Layer.size().");
        }
        return *(Layers[index]);
      }

      template <typename T>
      ILayer2D<T>& Network2D<T>::GetLayer(const size_t index)
      {
        if (index >= Layers.size())
        {
          throw std::range_error("cnn::engine::convolution::Network2D::GetLayer(), index >= Layer.size().");
        }
        return *(Layers[index]);
      }

      template <typename T>
      const ILayer2D<T>& Network2D<T>::GetLastLayer() const
      {
        if (Layers.size() == 0)
        {
          throw std::logic_error("cnn::engine::convolution::Network2D::GetLastLayer() const, Layers.size() == 0.");
        }
        return *(Layers.back());
      }

      template <typename T>
      ILayer2D<T>& Network2D<T>::GetLastLayer()
      {
        if (Layers.size() == 0)
        {
          throw std::logic_error("cnn::engine::convolution::Network2D::GetLastLayer(), Layers.size() == 0.");
        }
        return *(Layers.back());
      }

      template <typename T>
      const ILayer2D<T>& Network2D<T>::GetFirstLayer() const
      {
        if (Layers.size() == 0)
        {
          throw std::logic_error("cnn::engine::convolution::Network2D::GetFirstLayer() const, Layers.size() == 0.");
        }
        return *(Layers.front());
      }

      template <typename T>
      ILayer2D<T>& Network2D<T>::GetFirstLayer()
      {
        if (Layers.size() == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Network2D::GetFirstLayer(), Layers.size() == 0.");
        }
        return *(Layers.front());
      }

      template <typename T>
      void Network2D<T>::Process()
      {
        for (size_t l = 0; l < Layers.size(); ++l)
        {
          auto& layer = *(Layers[l]);
          if (l != 0)
          {
            const auto& prevLayer = *(Layers[l - 1]);
            for (size_t i = 0; i < layer.GetInputCount(); ++i)
            {
              const auto& prevOutput = prevLayer.GetOutput(i);
              auto& input = layer.GetInput(i);
              for (size_t x = 0; x < layer.GetInputWidth(); ++x)
              {
                for (size_t y = 0; y < layer.GetInputHeight(); ++y)
                {
                  const auto& value = prevOutput.GetValue(x, y);
                  input.SetValue(x, y, value);
                }
              }
            }
          }
          layer.Process();
        }
      }
    }
  }
}