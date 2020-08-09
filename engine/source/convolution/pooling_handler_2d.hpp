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
      PoolingHandler2D<T>::PoolingHandler2D(const size_t inputWidth,
                                            const size_t inputHeight,
                                            const size_t stepSize,// TODO: Change order of this and next value.
                                            const size_t channelCount)
        :
        InputWidth{ inputWidth },
        InputHeight{ inputHeight },
        StepSize{ stepSize },// TODO: Change order of this and next value.
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
        if (StepSize <= 1)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingHandler2D::PoolingHandler2D(), StepSize <= 1.");
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

        if (InputWidth % StepSize)
        {
          OutputWidth = InputWidth / StepSize + 1;
          if (OutputWidth == 0)
          {
            throw std::overflow_error("cnn::engine::convolution::PoolingHandler2D::PollingHandler2D(), OutputWidth was overflowed.");
          }
        } else {
          OutputWidth = InputWidth / StepSize;
        }
        
        if (InputHeight % StepSize)
        {
          OutputHeight = InputHeight / StepSize + 1;
          if (OutputHeight == 0)
          {
            throw std::overflow_error("cnn::engine::convolution::PoolingHandler2D::PollingHandler2D(), OutputHeight was overflowed.");
          }
        } else {
          OutputHeight = InputHeight / StepSize;
        }

        Inputs = std::make_unique<typename Map2D<T>::Uptr[]>(ChannelCount);
        for (size_t i = 0; i < ChannelCount; ++i)
        {
          Inputs[i] = std::make_unique<Map2D<T>>(InputWidth, InputHeight);
        }
        Outputs = std::make_unique<typename Map2D<T>::Uptr[]>(ChannelCount);
        for (size_t i = 0; i < ChannelCount; ++i)
        {
          Outputs[i] = std::make_unique<Map2D<T>>(OutputWidth, OutputHeight);
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
        for (size_t c = 0; c < ChannelCount; ++c)
        {
          auto& input = *(Inputs[c]);
          auto& output = *(Outputs[c]);

          for (size_t ox = 0; ox < OutputWidth; ++ox)
          {
            for (size_t oy = 0; oy < OutputHeight; ++oy)
            {
              T maxValue = std::numeric_limits<float>::min();// What about it?

              // TODO: Handle possible overflow (overflow is possible!).
              // Perhaps, we need check overflows in the constructor.

              const size_t ixBegin = ox * StepSize;
              const size_t ixEnd = ((ixBegin + StepSize) < InputWidth) ? (ixBegin + StepSize) : (InputWidth);

              const size_t iyBegin = oy * StepSize;
              const size_t iyEnd = ((iyBegin + StepSize) < InputHeight) ? (iyBegin + StepSize) : (InputHeight);

              for (size_t ix = ixBegin; ix < ixEnd; ++ix)
              {
                for (size_t ix = iyBegin; ix < iyEnd; ++ix)
                {
                  const T value = input.GetValue(ix, ix);
                  if (value > maxValue)
                  {
                    maxValue = value;
                  }
                }
              }

              output.SetValue(ox, oy, maxValue);
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