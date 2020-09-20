#pragma once

#include "i_layer_2d.hpp"
#include "map_2d.hpp"
#include "filter_2d.hpp"

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

        using Uptr = std::unique_ptr<Layer2D<T>>;

        Layer2D(const size_t inputWidth,
                const size_t inputHeight,
                const size_t inputCount,
                           
                const size_t filterWidth,
                const size_t filterHeight,
                const size_t filterCount);

        Layer2D(const ILayer2D<T>& prevLayer,
                const size_t filterWidth,
                const size_t filterHeight,
                const size_t filterCount);

        size_t GetInputWidth() const override;
        size_t GetInputHeight() const override;
        size_t GetInputCount() const override;
        const IMap2D<T>& GetInput(const size_t index) const override;
        IMap2D<T>& GetInput(const size_t index) override;

        size_t GetFilterWidth() const override;
        size_t GetFilterHeight() const override;
        size_t GetFilterCount() const override;
        IFilter2D<T>& GetFilter(const size_t index) override;
        const IFilter2D<T>& GetFilter(const size_t index) const override;

        size_t GetOutputWidth() const override;
        size_t GetOutputHeight() const override;
        size_t GetOutputCount() const override;
        const IMap2D<T>& GetOutput(const size_t index) const override;
        IMap2D<T>& GetOutput(const size_t index) override;

        void Process() override;

        size_t GetOutputValueCount() const override;

        void ClearInputs();
        void ClearFilters();
        void ClearOutputs();

        typename ILayer2D<T>::Uptr Clone(const bool cloneState) const override;

        Layer2D(const Layer2D<T>& layer,
                const bool cloneState);

        void FillWeights(common::IValueGenerator<T>& valueGenerator) override;

        void CrossFrom(const ILayer2D<T>& source1,
                       const ILayer2D<T>& source2) override;

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

        size_t OutputValueCount;

      };

      template <typename T>
      Layer2D<T>::Layer2D(const size_t inputWidth,
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
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Layer2D(), InputWidth == 0.");
        }
        if (InputHeight == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Layer2D(), InputHeight == 0.");
        }
        if (InputCount == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Layer2D(), InputCount == 0.");
        }

        if (FilterWidth == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Layer2D(), FilterWidth == 0.");
        }
        if (FilterWidth > InputWidth)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Layer2D(), FilterWidth > InputWidth.");
        }
        if (FilterHeight == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Layer2D(), FilterHeight == 0.");
        }
        if (FilterHeight > InputHeight)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Layer2D(), FilterHeight > InputHeight.");
        }
        if (FilterCount == 0)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Layer2D(), FilterCount == 0.");
        }

        // Maybe it is extra checks...
        if (OutputWidth > InputWidth)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Layer2D(), OutputWidth > InputWidth.");
        }
        if (OutputHeight > InputHeight)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Layer2D(), OutputHeight > InputHeight.");
        }

        {
          const size_t m1 = OutputWidth * OutputHeight;
          if ((m1 / OutputWidth) != OutputHeight)
          {
            throw std::overflow_error("cnn::engine::convolution::Layer2D::Layer2D(), (m1 / OutputWidth) != OutputHeight.");
          }
          const size_t m2 = m1 * OutputCount;
          if ((m2 / m1) != OutputCount)
          {
            throw std::overflow_error("cnn::engine::convolution::Layer2D::Layer2D(), (m2 / m1) != OutputCount.");
          }
          OutputValueCount = m2;
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
      Layer2D<T>::Layer2D(const ILayer2D<T>& prevLayer,
                          const size_t filterWidth,
                          const size_t filterHeight,
                          const size_t filterCount)
        :
        Layer2D{ prevLayer.GetOutputWidth(),
                 prevLayer.GetOutputHeight(),
                 prevLayer.GetOutputCount(),

                 filterWidth,
                 filterHeight,
                 filterCount }
      {
      }

      template <typename T>
      size_t Layer2D<T>::GetInputWidth() const
      {
        return InputWidth;
      }

      template <typename T>
      size_t Layer2D<T>::GetInputHeight() const
      {
        return InputHeight;
      }

      template <typename T>
      size_t Layer2D<T>::GetInputCount() const
      {
        return InputCount;
      }

      template <typename T>
      const IMap2D<T>& Layer2D<T>::GetInput(const size_t index) const
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::convolution::Layer2D::GetInput() const, index >= InputCount.");
        }
        return *(Inputs[index]);
      }

      template <typename T>
      IMap2D<T>& Layer2D<T>::GetInput(const size_t index)
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::convolution::Layer2D::GetInput(), index >= InputCount.");
        }
        return *(Inputs[index]);
      }

      template <typename T>
      size_t Layer2D<T>::GetFilterWidth() const
      {
        return FilterWidth;
      }

      template <typename T>
      size_t Layer2D<T>::GetFilterHeight() const
      {
        return FilterHeight;
      }

      template <typename T>
      size_t Layer2D<T>::GetFilterCount() const
      {
        return FilterCount;
      }

      template <typename T>
      const IFilter2D<T>& Layer2D<T>::GetFilter(const size_t index) const
      {
        if (index >= FilterCount)
        {
          throw std::range_error("cnn::engine::convolution::Layer2D::GetFilter() const, index >= FilterCount.");
        }
        return *(Filters[index]);
      }

      template <typename T>
      IFilter2D<T>& Layer2D<T>::GetFilter(const size_t index)
      {
        if (index >= FilterCount)
        {
          throw std::range_error("cnn::engine::convolution::Layer2D::GetFilter(), index >= FilterCount.");
        }
        return *(Filters[index]);
      }

      template <typename T>
      size_t Layer2D<T>::GetOutputWidth() const
      {
        return OutputWidth;
      }

      template <typename T>
      size_t Layer2D<T>::GetOutputHeight() const
      {
        return OutputHeight;
      }

      template <typename T>
      size_t Layer2D<T>::GetOutputCount() const
      {
        return OutputCount;
      }

      template <typename T>
      const IMap2D<T>& Layer2D<T>::GetOutput(const size_t index) const
      {
        if (index >= OutputCount)
        {
          throw std::range_error("cnn::engine::convolution::Layer2D::GetOutput() const, index >= OutputCount.");
        }
        return *(Outputs[index]);
      }

      template <typename T>
      IMap2D<T>& Layer2D<T>::GetOutput(const size_t index)
      {
        if (index >= OutputCount)
        {
          throw std::range_error("std::engine::convolution::Layer2D::GetOutput(), index >= OutputCount.");
        }
        return *(Outputs[index]);
      }

      template <typename T>
      void Layer2D<T>::Process()
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
      size_t Layer2D<T>::GetOutputValueCount() const
      {
        return OutputValueCount;
      }

      template <typename T>
      void Layer2D<T>::ClearInputs()
      {
        for (size_t i = 0; i < InputCount; ++i)
        {
          Inputs[i]->Clear();
        }
      }

      template <typename T>
      void Layer2D<T>::ClearFilters()
      {
        for (size_t f = 0; f < FilterCount; ++f)
        {
          Filters[f]->Clear();
        }
      }

      template <typename T>
      void Layer2D<T>::ClearOutputs()
      {
        for (size_t o = 0; o < OutputCount; ++o)
        {
          Outputs[o]->Clear();
        }
      }

      template <typename T>
      typename ILayer2D<T>::Uptr Layer2D<T>::Clone(const bool cloneState) const
      {
        return std::make_unique<Layer2D<T>>(*this, cloneState);
      }

      template <typename T>
      Layer2D<T>::Layer2D(const Layer2D<T>& layer, const bool cloneState)
        :
        InputWidth{ layer.GetInputWidth() },
        InputHeight{ layer.GetInputHeight() },
        InputCount{ layer.GetInputCount() },
        Inputs{ std::make_unique<typename IMap2D<T>::Uptr[]>(InputCount) },
        FilterWidth{ layer.GetFilterWidth() },
        FilterHeight{ layer.GetFilterHeight() },
        FilterCount{ layer.GetFilterCount() },
        Filters{ std::make_unique<typename IFilter2D<T>::Uptr[]>(FilterCount) },
        OutputWidth{ layer.GetOutputWidth() },
        OutputHeight{ layer.GetOutputHeight() },
        OutputCount{ layer.GetOutputCount() },
        Outputs{ std::make_unique<typename IMap2D<T>::Uptr[]>(OutputCount) },
        OutputValueCount{ layer.GetOutputValueCount() }
      {
        for (size_t i = 0; i < InputCount; ++i)
        {
          Inputs[i] = layer.GetInput(i).Clone(cloneState);
        }
        for (size_t f = 0; f < FilterCount; ++f)
        {
          Filters[f] = layer.GetFilter(f).Clone(cloneState);
        }
        for (size_t o = 0; o < OutputCount; ++o)
        {
          Outputs[o] = layer.GetOutput(o).Clone(cloneState);
        }
      }

      template <typename T>
      void Layer2D<T>::FillWeights(common::IValueGenerator<T>& valueGenerator)
      {
        for (size_t f = 0; f < FilterCount; ++f)
        {
          Filters[f]->FillWeights(valueGenerator);
        }
      }

      template <typename T>
      void Layer2D<T>::CrossFrom(const ILayer2D<T>& source1,
                                 const ILayer2D<T>& source2)
      {
        // Compare us with source1.
        {
          if (GetInputWidth() != source1.GetInputWidth())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetInputWidth() != source1.GetInputWidth().");
          }
          if (GetInputHeight() != source1.GetInputHeight())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetInputHeight() != source1.GetInputHeight().");
          }
          if (GetInputCount() != source1.GetInputCount())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetInputCount() != source1.GetInputCount().");
          }

          if (GetFilterWidth() != source1.GetFilterWidth())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetFilterWidth() != source1.GetFilterWidth().");
          }
          if (GetFilterHeight() != source1.GetFilterHeight())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetFilterHeight() != source1.GetFilterHeight().");
          }
          if (GetFilterCount() != source1.GetFilterCount())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetFilterCount() != source1.GetFilterCount().");
          }

          if (GetOutputWidth() != source1.GetOutputWidth())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetOutputWidth() != source1.GetOutputWidth().");
          }
          if (GetOutputHeight() != source1.GetOutputHeight())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetOutputHeight() != source1.GetOutputHeight().");
          }
          if (GetOutputCount() != source1.GetOutputCount())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetOutputCount() != source1.GetOutputCount().");
          }
        }

        // Compare us with source2.
        {
          if (GetInputWidth() != source2.GetInputWidth())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetInputWidth() != source2.GetInputWidth().");
          }
          if (GetInputHeight() != source2.GetInputHeight())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetInputHeight() != source2.GetInputHeight().");
          }
          if (GetInputCount() != source2.GetInputCount())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetInputCount() != source2.GetInputCount().");
          }

          if (GetFilterWidth() != source2.GetFilterWidth())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetFilterWidth() != source2.GetFilterWidth().");
          }
          if (GetFilterHeight() != source2.GetFilterHeight())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetFilterHeight() != source2.GetFilterHeight().");
          }
          if (GetFilterCount() != source2.GetFilterCount())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetFilterCount() != source2.GetFilterCount().");
          }

          if (GetOutputWidth() != source2.GetOutputWidth())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetOutputWidth() != source2.GetOutputWidth().");
          }
          if (GetOutputHeight() != source2.GetOutputHeight())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetOutputHeight() != source2.GetOutputHeight().");
          }
          if (GetOutputCount() != source2.GetOutputCount())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CrossFrom(), GetOutputCount() != source2.GetOutputCount().");
          }
        }

        for (size_t f = 0; f < GetFilterCount(); ++f)
        {
          Filters[f]->CrossFrom(source1.GetFilter(f), source2.GetFilter(f));
        }
      }
    }
  }
}