#pragma once

#include "i_layer_2d.hpp"
#include "pooling_handler_2d.hpp"
#include "convolution_handler_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Layer2D : public ILayer2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Layer2D(const size_t inputWidth,
                const size_t inputHeight,
                const size_t inputCount,
          
                const size_t poolingStepSize,
                
                const size_t convolutionFilterWidth,
                const size_t convolutionFilterHeight,
                const size_t convolutionFilterCount);

        IPoolingHandler2D<T>& GetPoolingHandler() override;
        const IPoolingHandler2D<T>& GetPoolingHandler() const override;

        IConvolutionHandler2D<T>& GetConvolutionHandler() override;
        const IConvolutionHandler2D<T>& GetConvolutionHandler() const override;

        void Process() override;

      private:

        typename IPoolingHandler2D<T>::Uptr PoolingHandler;
        typename IConvolutionHandler2D<T>::Uptr ConvolutionHandler;

      };

      template <typename T>
      Layer2D<T>::Layer2D(const size_t inputWidth,
                          const size_t inputHeight,
                          const size_t inputCount,

                          const size_t poolingStepSize,

                          const size_t convolutionFilterWidth,
                          const size_t convolutionFilterHeight,
                          const size_t convolutionFilterCount)
      {
        PoolingHandler = std::make_unique<PoolingHandler2D<T>>(inputWidth,
                                                               inputHeight,
                                                               poolingStepSize,
                                                               inputCount);

        ConvolutionHandler = std::make_unique<ConvolutionHandler2D<T>>(PoolingHandler->GetOutputWidth(),
                                                                       PoolingHandler->GetOutputHeight(),
                                                                       PoolingHandler->GetChannelCount(),
                                                                       convolutionFilterWidth,
                                                                       convolutionFilterHeight,
                                                                       convolutionFilterCount);
      }

      template <typename T>
      IPoolingHandler2D<T>& Layer2D<T>::GetPoolingHandler()
      {
        return *(PoolingHandler);
      }

      template <typename T>
      const IPoolingHandler2D<T>& Layer2D<T>::GetPoolingHandler() const
      {
        return *(PoolingHandler);
      }

      template <typename T>
      IConvolutionHandler2D<T>& Layer2D<T>::GetConvolutionHandler()
      {
        return *(ConvolutionHandler);
      }

      template <typename T>
      const IConvolutionHandler2D<T>& Layer2D<T>::GetConvolutionHandler() const
      {
        return *(ConvolutionHandler);
      }

      template <typename T>
      void Layer2D<T>::Process()
      {
        PoolingHandler->Process();
        ConvolutionHandler->Process();
      }
    }
  }
}