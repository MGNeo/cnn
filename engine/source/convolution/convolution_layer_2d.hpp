#pragma once

#include "i_layer_2d.hpp"
#include "convolution_handler_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class ConvolutionLayer2D : public ILayer2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ConvolutionLayer2D<T>>;

        ConvolutionLayer2D(const size_t inputWidth,
                           const size_t inputHeight,
                           const size_t inputCount,
                           
                           const size_t filterWidth,
                           const size_t filterHeight,
                           const size_t filterCount);

        ConvolutionLayer2D(const ILayer2D<T>& prevLayer,
                           const size_t filterWidth,
                           const size_t filterHeight,
                           const size_t filterCount);

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

        typename IConvolutionHandler2D<T>::Uptr ConvolutionHandler;

      };

      template <typename T>
      ConvolutionLayer2D<T>::ConvolutionLayer2D(const size_t inputWidth,
                                                const size_t inputHeight,
                                                const size_t inputCount,

                                                const size_t filterWidth,
                                                const size_t filterHeight,
                                                const size_t filterCount)
      {
        ConvolutionHandler = std::make_unique<ConvolutionHandler2D<T>>(inputWidth,
                                                                       inputHeight,
                                                                       inputCount,
                                                                       
                                                                       filterWidth,
                                                                       filterHeight,
                                                                       filterCount);
      }

      template <typename T>
      ConvolutionLayer2D<T>::ConvolutionLayer2D(const ILayer2D<T>& prevLayer,
                                                const size_t filterWidth,
                                                const size_t filterHeight,
                                                const size_t filterCount)
      {
        ConvolutionHandler = std::make_unique<ConvolutionHandler2D<T>>(prevLayer->GetOutputWidth(),
                                                                       prevLayer->GetOutputHeight(),
                                                                       prevLayer->GetOutputCount(),
          
                                                                       filterWidth,
                                                                       filterHeight,
                                                                       filterCount);
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetInputWidth() const
      {
        return ConvolutionHandler->GetInputWidth();
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetIntputHeight() const
      {
        return ConvolutionHandler->GetInputHeight();
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetInputCount() const
      {
        return ConvolutionHandler->GetInputCount();
      }

      template <typename T>
      const IMap2D<T>& ConvolutionLayer2D<T>::GetInput(const size_t index) const
      {
        return ConvolutionHandler->GetInput(index);
      }

      template <typename T>
      IMap2D<T>& ConvolutionLayer2D<T>::GetInput(const size_t index)
      {
        return ConvolutionHandler->GetInput(index);
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetOutputWidth() const
      {
        return ConvolutionHandler->GetOutputWidth();
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetOutputHeight() const
      {
        return ConvolutionHandler->GetOutputHeight();
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetOutputCount() const
      {
        return ConvolutionHandler->GetOutputCount();
      }

      template <typename T>
      const IMap2D<T>& ConvolutionLayer2D<T>::GetOutput(const size_t index) const
      {
        return ConvolutionHandler->GetOutput(index);
      }

      template <typename T>
      IMap2D<T>& ConvolutionLayer2D<T>::GetOutput(const size_t index)
      {
        return ConvolutionHandler->GetOutput(index);
      }

      template <typename T>
      void ConvolutionLayer2D<T>::Process()
      {
        ConvolutionHandler->Process();
      }
    }
  }
}