#pragma once

#include "i_layer_2d.hpp"
#include "map_2d.hpp"

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
        size_t GetInputHeight() const override;
        size_t GetInputCount() const override;

        const IMap2D<T>& GetInput(const size_t index) const override;
        IMap2D<T>& GetInput(const size_t index) override;

        size_t GetOutputWidth() const override;
        size_t GetOutputHeight() const override;
        size_t GetOutputCount() const override;

        const IMap2D<T>& GetOutput(const size_t index) const override;
        IMap2D<T>& GetOutput(const size_t index) override;

        void Process() override;

        void Accept(ILayer2DVisitor<T>& visitor) override;
        
        size_t GetOutputValueCount() const override;

        size_t GetStepSize() const;

      private:

        size_t InputWidth;
        size_t InputHeight;

        size_t ChannelCount;

        size_t OutputWidth;
        size_t OutputHeight;
        
        size_t OutputValueCount;

        size_t StepSize;

        std::unique_ptr<typename IMap2D<T>::Uptr[]> Inputs;
        std::unique_ptr<typename IMap2D<T>::Uptr[]> Outputs;

        void CheckExtendedOverflows() const;

      };

      template <typename T>
      PoolingLayer2D<T>::PoolingLayer2D(const size_t inputWidth,
                                        const size_t inputHeight,
                                        const size_t channelCount,
                                        const size_t stepSize)
        :
          InputWidth{ inputWidth },
          InputHeight{ inputHeight },
          ChannelCount{ channelCount },
          StepSize{ stepSize }
      {
        if (InputWidth == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingLayer2D::PoolingLayer2D(), InputWidth == 0.");
        }
        if (InputHeight == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingLayer2D::PoolingLayer2D(), InputHeight == 0.");
        }
        if (StepSize <= 1)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingLayer2D::PoolingLayer2D(), StepSize <= 1.");
        }
        if (StepSize > InputWidth)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingLayer2D::PoolingLayer2D(), StepSize > InputWidth.");
        }
        if (StepSize > InputHeight)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingLayer2D::PoolingLayer2D(), StepSize > InputHeight.");
        }
        if (ChannelCount == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::PoolingLayer2D::PoolingLayer2D(), ChannelCount == 0.");
        }

        if (InputWidth % StepSize)
        {
          OutputWidth = InputWidth / StepSize + 1;
          if (OutputWidth == 0)
          {
            throw std::overflow_error("cnn::engine::convolution::PoolingLayer2D::PoolingLayer2D(), OutputWidth was overflowed.");
          }
        } else {
          OutputWidth = InputWidth / StepSize;
        }

        if (InputHeight % StepSize)
        {
          OutputHeight = InputHeight / StepSize + 1;
          if (OutputHeight == 0)
          {
            throw std::overflow_error("cnn::engine::convolution::PoolingLayer2D::PoolingLayer2D(), OutputHeight was overflowed.");
          }
        } else {
          OutputHeight = InputHeight / StepSize;
        }

        {
          const size_t m1 = OutputWidth * OutputHeight;
          if ((m1 / OutputWidth) != OutputHeight)
          {
            throw std::overflow_error("cnn::engine::convolution::PoolingLayer2D::PoolingLayer2D(), (m1 / OutputWidth) != OutputHeight.");
          }
          const size_t m2 = m1 * ChannelCount;
          if ((m2 / m1) != ChannelCount)
          {
            throw std::overflow_error("cnn::engine::convolution::PoolingLayer2D::PoolingLayer2D(), (m2 / m1) != ChannelCount.");
          }
          OutputValueCount = m2;
        }

        CheckExtendedOverflows();

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

      // TODO: Change the signature to (6x size_t).
      template <typename T>
      PoolingLayer2D<T>::PoolingLayer2D(const ILayer2D<T>& prevLayer,
                                        const size_t stepSize)
        :
        PoolingLayer2D{ prevLayer.GetOutputWidth(),
                        prevLayer.GetOutputHeight(),
                        prevLayer.GetOutputCount(),
                        stepSize }
      {
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetInputWidth() const
      {
        return InputWidth;
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetInputHeight() const
      {
        return InputHeight;
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetInputCount() const
      {
        return ChannelCount;
      }

      template <typename T>
      const IMap2D<T>& PoolingLayer2D<T>::GetInput(const size_t index) const
      {
        if (index >= ChannelCount)
        {
          throw std::range_error("cnn::engine::convolution::PoolingLayer2D::GetInput() const, index >= ChannelCount.");
        }
        return *(Inputs[index]);
      }

      template <typename T>
      IMap2D<T>& PoolingLayer2D<T>::GetInput(const size_t index)
      {
        if (index >= ChannelCount)
        {
          throw std::range_error("cnn::engine::convolution::PoolingLayer2D::GetInput(), index >= ChannelCount.");
        }
        return *(Inputs[index]);
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetOutputWidth() const
      {
        return OutputWidth;
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetOutputHeight() const
      {
        return OutputHeight;
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetOutputCount() const
      {
        return ChannelCount;
      }

      template <typename T>
      const IMap2D<T>& PoolingLayer2D<T>::GetOutput(const size_t index) const
      {
        if (index >= ChannelCount)
        {
          throw std::range_error("cnn::engine::convolution::PoolingLayer2D::GetOutput() const, index >= ChannelCount.");
        }
        return *(Outputs[index]);
      }

      template <typename T>
      IMap2D<T>& PoolingLayer2D<T>::GetOutput(const size_t index)
      {
        if (index >= ChannelCount)
        {
          throw std::range_error("cnn::engine::convolution::PoolingLayer2D::GetOutput(), index >= ChannelCount.");
        }
        return *(Outputs[index]);
      }

      template <typename T>
      void PoolingLayer2D<T>::Process()
      {
        {
          for (size_t c = 0; c < ChannelCount; ++c)
          {
            auto& input = *(Inputs[c]);
            auto& output = *(Outputs[c]);

            for (size_t ox = 0; ox < OutputWidth; ++ox)
            {
              for (size_t oy = 0; oy < OutputHeight; ++oy)
              {
                T maxValue = std::numeric_limits<float>::min();

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
      }

      template <typename T>
      void PoolingLayer2D<T>::Accept(ILayer2DVisitor<T>& visitor)
      {
        visitor.Visit(*this);
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetOutputValueCount() const
      {
        return OutputValueCount;
      }

      template <typename T>
      size_t PoolingLayer2D<T>::GetStepSize() const
      {
        return StepSize;
      }

      template <typename T>
      void PoolingLayer2D<T>::CheckExtendedOverflows() const
      {
        {
          const size_t ixBegin = OutputWidth * StepSize;
          if ((ixBegin / OutputWidth) != StepSize)
          {
            throw std::overflow_error("cnn::engine::convolution::PoolingLayer2D::CheckExtendedOverflows(), ixBegin was overflowed.");
          }

          const size_t ixEnd = ixBegin + StepSize;
          if (ixEnd < ixBegin)
          {
            throw std::overflow_error("cnn::engine::convolution::PoolingLayer2D::CheckExtendedOverflows(), ixEnd was overflowed.");
          }
        }

        {
          const size_t iyBegin = OutputHeight * StepSize;
          if ((iyBegin / OutputHeight) != StepSize)
          {
            throw std::overflow_error("cnn::engine::convolution::PoolingLayer2D::CheckExtendedOverflows(), iyBegin was overflowed.");
          }

          const size_t iyEnd = iyBegin + StepSize;
          if (iyEnd < iyBegin)
          {
            throw std::overflow_error("cnn::engine::convolution::PoolingLayer2D::CheckExtendedOverflows(), iyEnd was overflowed.");
          }
        }
      }
    }
  }
}