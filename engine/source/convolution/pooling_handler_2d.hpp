#pragma once

#include <memory>
#include <type_traits>

#include "i_pooling_handler_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class PoolingHandler2D : public IPoolingHandler2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        PoolingHandler2D(const size_t inputWidth,
                         const size_t inputHeight,
                         const size_t stepSize,
                         const size_t channelCount);

        size_t GetInputWidth() const override;
        size_t GetInputHeight() const override;

        size_t GetStepSize() const override;

        size_t GetOutputWidth() const override;
        size_t GetOutputHeight() const override;

        size_t GetChannelCount() const override;

        IMap2D<T>& GetInput(const size_t index) override;
        const IMap2D<T>& GetInput(const size_t index) const override;

        IMap2D<T>& GetOutput(const size_t index) override;
        const IMap2D<T>& GetOutput(const size_t index) const override;

        void Process() override;

        void Clear() override;

      private:

        size_t InputWidth;
        size_t InputHeight;

        size_t StepSize;

        size_t OutputWidth;
        size_t OutputHeight;

        size_t ChannelCount;

        std::unique_ptr<typename IMap2D<T>::Uptr[]> Inputs;
        std::unique_ptr<typename IMap2D<T>::Uptr[]> Outputs;

      };

      template <typename T>
      PoolingHandler2D<T>::PoolingHandler2D(const size_t width,
                                            const size_t height,
                                            const size_t stepSize,
                                            const size_t channelCount)
        :
        InputWidth{ inputWidth },
        InputHeight{ inputHeight },
        StepSize{ stepSize },
        ChannelCount{ channelCount }
      {
        if (InputWidth == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingHandler2D::PoolingHandler2D(), InputWidth == 0.");
        }
        if (InputHeight == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingHandler2D::PoolingHandler2D(), InputHeight == 0.");
        }
        if (StepSize == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingHandler2D::PoolingHandler2D(), StepSize == 0.");
        }
        if (StepSize > InputWidth)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingHandler2D::PoolingHandler2D(), StepSize > InputWidth.");
        }
        if (StepSize > InputHeight)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingHandler2D::PoolingHandler2D(), StepSize > InputHeight.");
        }
        if (ChannelCount == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingHandler2D::PoolingHandler2D(), ChannelCount == 0.");
        }

        OutputWidth = InputWidth - StepSize + 1;
        if (OutputWidth == 0)
        {
          throw std::overflow_error("cnn::engine::convolution::PoolingHandler2D::PoolingHandler2D(), OutputWidth was overflowed.");
        }

        OutputHeight = InputHeight - StepSize + 1;
        if (OutputHeight == 0)
        {
          throw std::overflow_error("cnn::engine::convolution::PoolingHandler2D::PoolingHandler2D(), OutputHeight was overflowed.");
        }

        Inputs = std::make_unique<typename Map2D<T>::Uptr[]>(ChannelCount);
        for (size_t i = 0; i < ChannelCount; ++i)
        {
          Inputs[i] = std::make_unique<Map2D<T>>(Width, Height);
        }
        Outputs = std::make_unique<typename Map2D<T>::Uptr[]>(ChannelCount);
        for (size_t i = 0; i < ChannelCount; ++i)
        {
          Outputs[i] = std::make_unique<Map2D<T>>(Width, Height);
        }
      }

      template <typename T>
      size_t PoolingHandler2D<T>::GetInputWidth() const
      {
        return InputWidth;
      }

      template <typename T>
      size_t PoolingHandler2D<T>::GetInputHeight() const
      {
        return InputHeight;
      }

      template <typename T>
      size_t PoolingHandler2D<T>::GetStepSize() const
      {
        return StepSize;
      }

      template <typename T>
      size_t PoolingHandler2D<T>::GetOutputWidth() const
      {
        return OutputWidth;
      }

      template <typename T>
      size_t PoolingHandler2D<T>::GetOutputHeight() const
      {
        return OutputHeight;
      }

      template <typename T>
      size_t PoolingHandler2D<T>::GetChannelCount() const
      {
        return ChannelCount;
      }

      template <typename T>
      IMap2D<T>& PoolingHandler2D<T>::GetInput(const size_t index)
      {
        if (index >= ChannelCount)
        {
          throw std::range_error("cnn::engine::convolution::PoolingHandler2D::GetInput(), index >= ChannelCount.");
        }
        return *(Inputs[index]);
      }

      template <typename T>
      const IMap2D<T>& PoolingHandler2D<T>::GetInput(const size_t index) const
      {
        if (index >= ChannelCount)
        {
          throw std::range_error("cnn::engine::convolution::PoolingHandler2D::GetInput() const, index >= ChannelCount.");
        }
        return *(Inputs[index]);
      }

      template <typename T>
      IMap2D<T>& PoolingHandler2D<T>::GetOutput(const size_t index)
      {
        if (index >= ChannelCount)
        {
          throw std::range_error("cnn::engine::convolution::PoolingHandler2D::GetOutput(), index >= ChannelCount.");
        }
        return *(Outputs[index]);
      }

      template <typename T>
      const IMap2D<T>& PoolingHandler2D<T>::GetOutput(const size_t index) const
      {
        if (index >= ChannelCount)
        {
          throw std::range_error("cnn::engine::convolution::PoolingHandler2D::GetOutput() const, index >= ChannelCount.");
        }
        return *(Outputs[index]);
      }

      template <typename T>
      void PoolingHandler2D<T>::Process()
      {
        for (size_t i = 0; i < ChannelCount; ++i)
        {
          auto& input = *(Inputs[i]);
          auto& output = *(Outputs[i]);
          for (size_t x = 0; x < InputWidth; x += StepSize)
          {
            for (size_t y = 0; y < InputHeight; y += StepSize)
            {
              const T maxValue = std::numeric_limits<float>::min();// What about it?
              
              const size_t fromX = x;
              const size_t toX = ((x + StepSize) <= InputWidth) ? (x + StepSize) : InpuwWidth;

              const size_t fromY = y;
              const size_t toY = ((y + StepSize) <= InputHeight) ? (y + StepSize) : InputHeight;

              for (size_t localX = fromX; localX < toX; ++localX)
              {
                for (size_t localY = fromY; localY < toY; ++localY)
                {
                  const T value = input.GetValue(localX, localY);
                  if (value > maxValue)
                  {
                    maxValue = value;
                  }
                }
              }

              output.SetValue(x, y, maxValue);
            }
          }
        }
      }

      template <typename T>
      void PoolingHandler2D<T>::Clear()
      {
        for (size_t i = 0; i < ChannelCount; ++i)
        {
          Inputs[i]->Clear();
          Outputs[i]->Clear();
        }
      }
    }
  }
}