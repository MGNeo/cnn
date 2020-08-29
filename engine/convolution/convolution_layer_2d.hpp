#pragma once

#include "i_layer_2d.hpp"
#include "map_2d.hpp"
#include "filter_2d.hpp"

#include "i_layer_2d_visitor.hpp"

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
        size_t GetInputHeight() const override;
        size_t GetInputCount() const override;

        const IMap2D<T>& GetInput(const size_t index) const override;
        IMap2D<T>& GetInput(const size_t index) override;

        size_t GetFilterWidth() const;
        size_t GetFilterHeight() const;
        size_t GetFilterCount() const;

        IFilter2D<T>& GetFilter(const size_t index);
        const IFilter2D<T>& GetFilter(const size_t index) const;

        size_t GetOutputWidth() const override;
        size_t GetOutputHeight() const override;
        size_t GetOutputCount() const override;

        const IMap2D<T>& GetOutput(const size_t index) const override;
        IMap2D<T>& GetOutput(const size_t index) override;

        void Process() override;

        void Accept(ILayer2DVisitor<T>& visitor) override;

        void ClearInputs();
        void ClearFilters();
        void ClearOutputs();

      private:

        size_t InputWidth;
        size_t InputHeight;
        size_t InputCount;
        std::unique_ptr<typename IMap2D<T>::Uptr[]> Inputs;

        size_t FilterWidth;
        size_t FilterHeight;
        size_t FilterCount;
        std::unique_ptr<typename IFilter2D<T>::Uptr[]> Filters;

        size_t OutputWidth;
        size_t OutputHeight;
        size_t OutputCount;
        std::unique_ptr<typename IMap2D<T>::Uptr[]> Outputs;

      };

      template <typename T>
      ConvolutionLayer2D<T>::ConvolutionLayer2D(const size_t inputWidth,
                                                const size_t inputHeight,
                                                const size_t inputCount,

                                                const size_t filterWidth,
                                                const size_t filterHeight,
                                                const size_t filterCount)
        :
        InputWidth{ inputWidth },
        InputHeight{ inputHeight },
        InputCount{ inputCount },

        FilterWidth{ filterWidth },
        FilterHeight{ filterHeight },
        FilterCount{ filterCount },

        OutputWidth{ InputWidth - FilterWidth + 1 },
        OutputHeight{ InputHeight - FilterHeight + 1 },
        OutputCount{ FilterCount }
      {
        if (InputWidth == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::ConvolutionLayer2D::ConvolutionLayer2D(), InputWidth == 0.");
        }
        if (InputHeight == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::ConvolutionLayer2D::ConvolutionLayer2D(), InputHeight == 0.");
        }
        if (InputCount == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::ConvolutionLayer2D::ConvolutionLayer2D(), InputCount == 0.");
        }

        if (FilterWidth == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::ConvolutionLayer2D::ConvolutionLayer2D(), FilterWidth == 0.");
        }
        if (FilterWidth > InputWidth)
        {
          throw std::invalid_argument("cnn::engine::convolution::ConvolutionLayer2D::ConvolutionLayer2D(), FilterWidth > InputWidth.");
        }
        if (FilterHeight == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::ConvolutionLayer2D::ConvolutionLayer2D(), FilterHeight == 0.");
        }
        if (FilterHeight > InputHeight)
        {
          throw std::invalid_argument("cnn::engine::convolution::ConvolutionLayer2D::ConvolutionLayer2D(), FilterHeight > InputHeight.");
        }
        if (FilterCount == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::ConvolutionLayer2D::ConvolutionLayer2D(), FilterCount == 0.");
        }

        if (OutputWidth > InputWidth)
        {
          throw std::invalid_argument("cnn::engine::convolution::ConvolutionHandler2D::ConvolutionHandler2D(), OutputWidth > InputWidth.");
        }
        if (OutputHeight > InputHeight)
        {
          throw std::invalid_argument("cnn::engine::convolution::ConvolutionHandler2D::ConvolutionHandler2D(), OutputHeight > InputHeight.");
        }

        Inputs = std::make_unique<typename IMap2D<T>::Uptr[]>(InputCount);
        for (size_t i = 0; i < InputCount; ++i)
        {
          Inputs[i] = std::make_unique<Map2D<T>>(InputWidth, InputHeight);
        }
        ClearInputs();

        Filters = std::make_unique<typename IFilter2D<T>::Uptr[]>(FilterCount);
        for (size_t f = 0; f < FilterCount; ++f)
        {
          Filters[f] = std::make_unique<Filter2D<T>>(FilterWidth, FilterHeight, InputCount);
        }
        ClearFilters();

        Outputs = std::make_unique<typename IMap2D<T>::Uptr[]>(OutputCount);
        for (size_t o = 0; o < OutputCount; ++o)
        {
          Outputs[o] = std::make_unique<Map2D<T>>(OutputWidth, OutputHeight);
        }
        ClearOutputs();
      }

      // TODO: Change the signature to (6x size_t).
      template <typename T>
      ConvolutionLayer2D<T>::ConvolutionLayer2D(const ILayer2D<T>& prevLayer,
                                                const size_t filterWidth,
                                                const size_t filterHeight,
                                                const size_t filterCount)
        :
        ConvolutionLayer2D{ prevLayer.GetOutputWidth(),
                            prevLayer.GetOutputHeight(),
                            prevLayer.GetOutputCount(),

                            filterWidth,
                            filterHeight,
                            filterCount }
      {
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetInputWidth() const
      {
        return InputWidth;
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetInputHeight() const
      {
        return InputHeight;
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetInputCount() const
      {
        return InputCount;
      }

      template <typename T>
      const IMap2D<T>& ConvolutionLayer2D<T>::GetInput(const size_t index) const
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::convolution::ConvolutionLayer2D::GetInput() const, index >= InputCount.");
        }
        return *(Inputs[index]);
      }

      template <typename T>
      IMap2D<T>& ConvolutionLayer2D<T>::GetInput(const size_t index)
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::convolution::ConvolutionHandler2D::GetInput(), index >= InputCount.");
        }
        return *(Inputs[index]);
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetFilterWidth() const
      {
        return FilterWidth;
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetFilterHeight() const
      {
        return FilterHeight;
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetFilterCount() const
      {
        return FilterCount;
      }

      template <typename T>
      const IFilter2D<T>& ConvolutionLayer2D<T>::GetFilter(const size_t index) const
      {
        if (index >= FilterCount)
        {
          throw std::range_error("cnn::engine::convolution::ConvolutionLayer2D::GetFilter() const, index >= FilterCount.");
        }
        return *(Filters[index]);
      }

      template <typename T>
      IFilter2D<T>& ConvolutionLayer2D<T>::GetFilter(const size_t index)
      {
        if (index >= FilterCount)
        {
          throw std::range_error("cnn::engine::convolution::ConvolutionLayer2D::GetFilter(), index >= FilterCount.");
        }
        return *(Filters[index]);
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetOutputWidth() const
      {
        return OutputWidth;
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetOutputHeight() const
      {
        return OutputHeight;
      }

      template <typename T>
      size_t ConvolutionLayer2D<T>::GetOutputCount() const
      {
        return OutputCount;
      }

      template <typename T>
      const IMap2D<T>& ConvolutionLayer2D<T>::GetOutput(const size_t index) const
      {
        if (index >= OutputCount)
        {
          throw std::range_error("cnn::engine::convolution::ConvolutionLayer2D::GetOutput() const, index >= OutputCount.");
        }
        return *(Outputs[index]);
      }

      template <typename T>
      IMap2D<T>& ConvolutionLayer2D<T>::GetOutput(const size_t index)
      {
        if (index >= OutputCount)
        {
          throw std::range_error("std::engine::convolution::ConvolutionLayer2D::GetOutput(), index >= OutputCount.");
        }
        return *(Outputs[index]);
      }

      template <typename T>
      void ConvolutionLayer2D<T>::Process()
      {
        // TODO: Think about rollback when exception is thrown.
        for (size_t f = 0; f < FilterCount; ++f)
        {
          auto& filter = *(Filters[f]);
          auto& output = *(Outputs[f]);
          output.Clear();
          for (size_t i = 0; i < InputCount; ++i)
          {
            const auto& input = *(Inputs[i]);
            auto& core = filter.GetCore(i);
            core.ClearInputs();
            for (size_t ox = 0; ox < OutputWidth; ++ox)
            {
              for (size_t oy = 0; oy < OutputHeight; ++oy)
              {
                for (size_t cx = 0; cx < FilterWidth; ++cx)
                {
                  for (size_t cy = 0; cy < FilterHeight; ++cy)
                  {
                    const auto ix = ox + cx;
                    const auto iy = oy + cy;
                    const auto value = input.GetValue(ix, iy);
                    core.SetInput(cx, cy, value);
                  }
                }
                core.Process();
                const auto value = output.GetValue(ox, oy) + core.GetOutput();
                output.SetValue(ox, oy, value);
              }
            }
          }
        }
      }

      template <typename T>
      void ConvolutionLayer2D<T>::Accept(ILayer2DVisitor<T>& visitor)
      {
        visitor.Visit(*this);
      }

      template <typename T>
      void ConvolutionLayer2D<T>::ClearInputs()
      {
        for (size_t i = 0; i < InputCount; ++i)
        {
          Inputs[i]->Clear();
        }
      }

      template <typename T>
      void ConvolutionLayer2D<T>::ClearFilters()
      {
        for (size_t f = 0; f < FilterCount; ++f)
        {
          Filters[f]->Clear();
        }
      }

      template <typename T>
      void ConvolutionLayer2D<T>::ClearOutputs()
      {
        for (size_t o = 0; o < OutputCount; ++o)
        {
          Outputs[o]->Clear();
        }
      }
    }
  }
}