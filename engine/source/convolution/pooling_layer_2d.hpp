#pragma once

#include "i_layer_2d.hpp"
#include "pooling_handler_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class PoolingLayer2D : public ILayer2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<PoolingLayer2D<T>>;

        PoolingLayer2D(const size_t inputWidth,
                       const size_t inputHeight,
                       const size_t channelCount,
                       const size_t stepSize);

        PoolingLayer2D(const ILayer2D<T>& prevLayer,
                       const size_t stepSize);

        size_t GetInputWidth() const override;
        size_t GetIntputHeight() const override;
        size_t GetInputCount() const override;

        const IMap2D<T>& GetInput(const size_t index) const override;
        IMap2D<T>& GetInput(const size_t index) override;

        size_t GetOutputWidth() const override;
        size_t GetOutputHeight() const override;
        size_t GetOutputCount() const override;

        const IMap2D<T>& GetOutput(const size_t index) const override;
        IMap2D<T>& GetOutput(const size_t index) override;

        void Process() override;

      private:

        typename IPoolingHandler2D<T>::Uptr PoolingHandler;

      };

      template <typename T>
      PoolingLayer2D<T>::PoolingLayer2D(const size_t inputWidth,
                                        const size_t inputHeight,
                                        const size_t channelCount,
                                        const size_t stepSize)
      {
        PoolingHandler = std::make_unique<PoolingHandler2D<T>>(inputWidth,
                                                               inputHeight,
                                                               stepSize,
                                                               channelCount);
      }

      template <typename T>
      PoolingLayer2D<T>::PoolingLayer2D(const ILayer2D<T>& prevLayer,
                                        const size_t stepSize)
      {
        PoolingHandler = std::make_unique<PoolingHandler2D<T>>(prevLayer->GetOutputWidth(),
                                                               prevLayer->GetOutputHeight(),
                                                               stepSize,
                                                               prevLayer->GetOutputCount());
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetInputWidth() const
      {
        return PoolingHandler->GetInputWidth();
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetIntputHeight() const
      {
        return PoolingHandler->GetInputHeight();
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetInputCount() const
      {
        return PoolingHandler->GetChannelCount();
      }

      template <typename T>
      const IMap2D<T>& PoolingLayer2D<T>::GetInput(const size_t index) const
      {
        return PoolingHandler->GetInput(index);
      }

      template <typename T>
      IMap2D<T>& PoolingLayer2D<T>::GetInput(const size_t index)
      {
        return PoolingHandler->GetInput(index);
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetOutputWidth() const
      {
        return PoolingHandler->GetOutputWidth();
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetOutputHeight() const
      {
        return PoolingHandler->GetOutputHeight();
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetOutputCount() const
      {
        return PoolingHandler->GetChannelCount();
      }

      template <typename T>
      const IMap2D<T>& PoolingLayer2D<T>::GetOutput(const size_t index) const
      {
        return PoolingHandler->GetOutput(index);
      }

      template <typename T>
      IMap2D<T>& PoolingLayer2D<T>::GetOutput(const size_t index)
      {
        return PoolingHandler->GetOutput(index);
      }

      template <typename T>
      void PoolingLayer2D<T>::Process()
      {
        PoolingHandler->Process();
      }
    }
  }
}