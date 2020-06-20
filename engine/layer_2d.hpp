#pragma once

#include <stdexcept>

#include "i_layer_2d.hpp"
#include "map_2d.hpp"
#include "filter_2d.hpp"

namespace cnn
{
  template <typename T>
  class Layer2D : public ILayer2D<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    // TODO: Add special flag (UsePadding).
    Layer2D(const size_t inputCount,
            const size_t inputWidth,
            const size_t inputHeight,

            const size_t filterCount,
            const size_t filterWidth,
            const size_t filterHeight,
      
            typename const IActivator<T>& activator);

    size_t GetInputCount() const override;
    size_t GetInputWidth() const override;
    size_t GetInputHeight() const override;

    size_t GetFilterCount() const override;
    size_t GetFilterWidth() const override;
    size_t GetFilterHeight() const override;

    size_t GetOutputCount() const override;
    size_t GetOutputWidth() const override;
    size_t GetOutputHeight() const override;

    // TODO: const IActivator<T>& GetActivator() const override;

    const IMap2D<T>& GetInput(const size_t index) const override;
    IMap2D<T>& GetInput(const size_t index) override;

    const IFilter2D<T>& GetFilter(const size_t index) const override;
    IFilter2D<T>& GetFilter(const size_t index) override;

    const IMap2D<T>& GetOutput(const size_t index) const override;
    IMap2D<T>& GetOutput(const size_t index) override;

    void Process() const override;

  private:

    const size_t InputCount;
    const size_t InputWidth;
    const size_t InputHeight;
    std::unique_ptr<typename IMap2D<T>::Uptr[]> Inputs;

    const size_t FilterCount;
    const size_t FilterWidth;
    const size_t FilterHeight;
    std::unique_ptr<typename IFilter2D<T>::Uptr[]> Filters;

    const size_t OutputCount;
    const size_t OutputWidth;
    const size_t OutputHeight;
    std::unique_ptr<typename IMap2D<T>::Uptr[]> Outputs;

    typename const IActivator<T>& Activator;

  };

  template <typename T>
  Layer2D<T>::Layer2D(const size_t inputCount,
                      const size_t inputWidth,
                      const size_t inputHeight,

                      const size_t filterCount,
                      const size_t filterWidth,
                      const size_t filterHeight,
    
                      typename const IActivator<T>& activator)
    :
    InputCount{ inputCount },
    InputWidth{ inputWidth },
    InputHeight{ inputHeight },
    Inputs{ std::make_unique<typename IMap2D<T>::Uptr[]>(InputCount) },

    FilterCount{ filterCount },
    FilterWidth{ filterWidth },
    FilterHeight{ filterHeight },
    Filters{ std::make_unique<typename IFilter2D<T>::Uptr[]>(FilterCount) },

    OutputCount{ filterCount },
    OutputWidth{ InputWidth - FilterWidth + 1 },
    OutputHeight{ InputHeight - FilterHeight + 1 },
    Outputs{ std::make_unique<typename IMap2D<T>::Uptr[]>(OutputCount) },

    Activator{ activator }
  {
    if (InputCount == 0)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), InputCount == 0.");
    }
    if (InputWidth == 0)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), InputWidth == 0.");
    }
    if (InputHeight == 0)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), InputHeight == 0.");
    }

    if (FilterCount == 0)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), FilterCount == 0.");
    }
    if (FilterWidth == 0)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), FilterWidth == 0.");
    }
    if (FilterHeight == 0)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), FilterHeight == 0.");
    }
    if (FilterWidth > InputWidth)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), FilterWidth > InputWIdth.");
    }
    if (FilterHeight > InputHeight)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), FilterHeight > InputHeight.");
    }
    if (FilterWidth % 2 == 0)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), FilterWidth % 2 == 0.");
    }
    if (FilterHeight % 2 == 0)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), FilterHeight % 2 == 0.");
    }

    if (OutputCount == 0)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), OutputCount == 0.");
    }
    if (OutputWidth == 0)
    {
      throw std::invalid_argument("cnn""Layer2D::Layer2D(), OutputWidth == 0.");
    }
    if (OutputHeight == 0)
    {
      throw std::invalid_argument("cnn::Layer2D::Layer2D(), OutputHeight == 0.");
    }

    for (size_t i = 0; i < InputCount; ++i)
    {
      Inputs[i] = std::make_unique<Map2D<T>>(InputWidth, InputHeight);
    }

    for (size_t i = 0; i < FilterCount; ++i)
    {
      Filters[i] = std::make_unique<Filter2D<T>>(FilterCount, FilterWidth, FilterHeight);
    }

    for (size_t i = 0; i < OutputCount; ++i)
    {
      Outputs[i] = std::make_unique<Map2D<T>>(OutputWidth, OutputHeight);
    }
  }

  template <typename T>
  size_t Layer2D<T>::GetInputCount() const
  {
    return InputCount;
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
  size_t Layer2D<T>::GetFilterCount() const
  {
    return FilterCount;
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
  size_t Layer2D<T>::GetOutputCount() const
  {
    return OutputCount;
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
  const IMap2D<T>& Layer2D<T>::GetInput(const size_t index) const
  {
    if (index >= InputCount)
    {
      throw std::range_error("cnn::Layer2D::GetInput() const, index >= InputCount.");
    }
    return *(Inputs[index]);
  }

  template <typename T>
  IMap2D<T>& Layer2D<T>::GetInput(const size_t index)
  {
    if (index >= InputCount)
    {
      throw std::range_error("cnn::Layer2D::GetInput(), index >= InputCount.");
    }
    return *(Inputs[index]);
  }

  template <typename T>
  const IFilter2D<T>& Layer2D<T>::GetFilter(const size_t index) const
  {
    if (index >= FilterCount)
    {
      throw std::range_error("cnn::Layer2D::GetFilter() const, index >= FilterCount.");
    }
    return *(Filters[index]);
  }

  template <typename T>
  IFilter2D<T>& Layer2D<T>::GetFilter(const size_t index)
  {
    if (index >= FilterCount)
    {
      throw std::range_error("cnn::Layer2D::GetFilter(), index >= FilterCount.");
    }
    return *(Filters[index]);
  }

  template <typename T>
  const IMap2D<T>& Layer2D<T>::GetOutput(const size_t index) const
  {
    if (index >= OutputCount)
    {
      throw std::range_error("cnn::Layer2D::GetOutput() const, index >= OutputCOunt.");
    }
    return *(Outputs[index]);
  }

  template <typename T>
  IMap2D<T>& Layer2D<T>::GetOutput(const size_t index)
  {
    if (index >= OutputCount)
    {
      throw std::range_error("cnn::Layer2D::GetOutput(), index >= OutputCount.");
    }
    return *(Outputs[index]);
  }

  template <typename T>
  void Layer2D<T>::Process() const
  {
    // Perhaps, we need to use a GPU instead of a CPU...
    for (size_t f = 0; f < FilterCount; ++f)
    {
      auto& filter = Filters[f];
      auto& output = Outputs[f];
      // TODO: First of all (ALL!), we must clear all outputs (for exception safety).
      output->Clear();
      for (size_t i = 0; i < InputCount; ++i)
      {
        auto& input = Inputs[i];
        auto& core = filter->GetCore(i);
        for (size_t ox = 0; ox < OutputWidth; ++ox)
        {
          for (size_t oy = 0; oy < OutputHeight; ++oy)
          {
            for (size_t fx = 0; fx < FilterWidth; ++fx)
            {
              for (size_t fy = 0; fy < FilterHeight; ++fy)
              {
                // Yes, Standard allows us to do it.
                const T& value = input->GetValue(ox + fx, oy + fy);
                core.SetInput(fx, fy, value);
              }
            }
            core.GenerateOutput();
            const T value = output->GetValue(ox, oy) + core.GetOutput();
            output->SetValue(ox, oy, value);
          }
        }
      }
      for (size_t ox = 0; ox < OutputWidth; ++ox)
      {
        for (size_t oy = 0; oy < OutputHeight; ++oy)
        {
          const T value = Activator.Handle(output->GetValue(ox, oy));
          output->SetValue(ox, oy, value);
        }
      }
    }
  }
}