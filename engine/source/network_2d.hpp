#pragma once

#include <vector>

#include "i_network_2d.hpp"
#include "layer_2d.hpp"
#include "activator.hpp"

namespace cnn
{
  // TODO: High level type must have default type argument (float).
  template <typename T>
  class Network2D : public INetwork2D<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Network2D(const size_t inputCount,
              const size_t inputWidth,
              const size_t inputHeight);

    size_t GetInputCount() const override;
    size_t GetInputWidth() const override;
    size_t GetInputHeight() const override;

    const IMap2D<T>& GetInput(const size_t index) const override;
    IMap2D<T>& GetInput(const size_t index) override;

    size_t GetLayerCount() const override;

    const ILayer2D<T>& GetLayer(const size_t index) const override;
    ILayer2D<T>& GetLayer(const size_t index) override;

    const ILayer2D<T>& GetLastLayer() const override;
    ILayer2D<T>& GetLastLayer() override;

    void PushLayer(const size_t filterCount,
                   const size_t filterWidth,
                   const size_t filterHeight,
                   typename const IActivator<T>& activator) override;

    void Process() override;

  private:

    const size_t InputCount;
    const size_t InputWidth;
    const size_t InputHeight;
    const std::unique_ptr<typename IMap2D<T>::Uptr[]> Inputs;

    std::vector<typename ILayer2D<T>::Uptr> Layers;

  };

  template <typename T>
  Network2D<T>::Network2D(const size_t inputCount,
                          const size_t inputWidth,
                          const size_t inputHeight)
    :
    InputCount{ inputCount },
    InputWidth{ inputWidth },
    InputHeight{ inputHeight },
    Inputs{ std::make_unique<typename IMap2D<T>::Uptr[]>(InputCount) }
  {
    if (InputCount == 0)
    {
      throw std::invalid_argument("cnn::Network2D::Network2D(), InputCount == 0.");
    }
    if (InputWidth == 0)
    {
      throw std::invalid_argument("cnn::Network2D::Network2D(), InputWidth == 0.");
    }
    if (InputHeight == 0)
    {
      throw std::invalid_argument("cnn::Network2D::Network2D(), InputHeight == 0.");
    }
    for (size_t i = 0; i < InputCount; ++i)
    {
      Inputs[i] = std::make_unique<Map2D<T>>(InputWidth, InputHeight);
    }
    // ...
  }

  template <typename T>
  size_t Network2D<T>::GetInputCount() const
  {
    return InputCount;
  }

  template <typename T>
  size_t Network2D<T>::GetInputWidth() const
  {
    return InputWidth;
  }

  template <typename T>
  size_t Network2D<T>::GetInputHeight() const
  {
    return InputHeight;
  }

  template <typename T>
  const IMap2D<T>& Network2D<T>::GetInput(const size_t index) const
  {
    if (index >= InputCount)
    {
      throw std::invalid_argument("cnn::Network2D::GetInput() const, index >= InputCount.");
    }
    return *(Inputs[index]);
  }

  template <typename T>
  IMap2D<T>& Network2D<T>::GetInput(const size_t index)
  {
    if (index >= InputCount)
    {
      throw std::invalid_argument("cnn::Network2D::GetInput(), index >= InputCount.");
    }
    return *(Inputs[index]);
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
      throw std::range_error("cnn::Network2D::GetLayer() const, index >= Layers.size().");
    }
    return *(Layers[index]);
  }

  template <typename T>
  ILayer2D<T>& Network2D<T>::GetLayer(const size_t index)
  {
    if (index >= Layers.size())
    {
      throw std::range_error("cnn::Network2D::GetLayer(), index >= Layers.size().");
    }
    return *(Layers[index]);
  }

  template <typename T>
  const ILayer2D<T>& Network2D<T>::GetLastLayer() const
  {
    if (Layers.size() == 0)
    {
      throw std::logic_error("cnn::Network2D::GetLastLayer() const, Layers.size() == 0.");
    }
    return *(Layers.back());
  }

  template <typename T>
  ILayer2D<T>& Network2D<T>::GetLastLayer()
  {
    if (Layers.size() == 0)
    {
      throw std::logic_error("cnn::Network2D::GetLastLayer(), Layers.size() == 0.");
    }
    return *(Layers.back());
  }

  template <typename T>
  void Network2D<T>::PushLayer(const size_t filterCount,
                               const size_t filterWidth,
                               const size_t filterHeight,
                               typename const IActivator<T>& activator)
  {
    if (filterCount == 0)
    {
      throw std::invalid_argument("cnn::Network2D::PushLayer(), filterCount == 0.");
    }
    if (filterWidth == 0)
    {
      throw std::invalid_argument("cnn::Network2D::PushLayer(), filterWidth == 0.");
    }
    if (filterHeight == 0)
    {
      throw std::invalid_argument("cnn::Network2D::PushLayer(), filterHeight == 0.");
    }
    if (Layers.size() == 0)
    {
      if (filterWidth > InputWidth)
      {
        throw std::invalid_argument("cnn::Network2D::PushLayer(), filterWidth > InputWidth.");
      }
      if (filterHeight > InputHeight)
      {
        throw std::invalid_argument("cnn::Network2D::PushLayer(), filterHeight > InputHeight.");
      }
    } else {
      if (filterWidth > Layers.back()->GetOutputWidth())
      {
        throw std::invalid_argument("cnn::Network2D::PushLayer(), filterWidth > Layers.back()->GetOutputWidth().");
      }
      if (filterHeight > Layers.back()->GetOutputHeight())
      {
        throw std::invalid_argument("cnn::Network2D::PushLayer(), filterHeight > Layers.back()->GetOutputHeight().");
      }
    }

    size_t inputCount{};
    size_t inputWidth{};
    size_t inputHeight{};

    if (Layers.size() == 0)
    {
      inputCount = InputCount;
      inputWidth = InputWidth;
      inputHeight = InputHeight;
    } else {
      inputCount = Layers.back()->GetOutputCount();
      inputWidth = Layers.back()->GetOutputWidth();
      inputHeight = Layers.back()->GetOutputHeight();
    }

    typename ILayer2D<T>::Uptr layer_2d = std::make_unique<Layer2D<T>>(inputCount,
                                                                       inputWidth,
                                                                       inputHeight,
                                                                       filterCount,
                                                                       filterWidth,
                                                                       filterHeight,
                                                                       activator);
    Layers.push_back(std::move(layer_2d));
  }

  template <typename T>
  void Network2D<T>::Process()
  {
    if (Layers.size() == 0)
    {
      return;
    }
    for (size_t i = 0; i < InputCount; ++i)
    {
      const auto& prev = *(Inputs[i]);
      auto& current = Layers.front()->GetInput(i);
      current.Copy(prev);
    }
    Layers.front()->Process();
    for (size_t l = 1; l < Layers.size(); ++l)
    {
      for (size_t i = 0; i < Layers[l]->GetInputCount(); ++i)
      {
        const auto& prev = Layers[l - 1]->GetOutput(i);
        auto& current = Layers[l]->GetInput(i);
        current.Copy(prev);
      }
      Layers[l]->Process();
    }
  }
}