#pragma once

#include <vector>

#include "i_perceptron.hpp"
#include "layer.hpp"

namespace cnn
{
  /*
  template <typename T>
  class Perceptron
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    Perceptron();

    size_t GetLayerCount() const override;

    const ILayer<T>& GetLayer(const size_t index) const override;
    ILayer<T>& GetLayer(const size_t index) override;

    void PushLayer(const size_t count) override;

    void Process() override;

  private:

    std::vector<typename ILayer<T>::Uptr> Layers;

  };
  */
}