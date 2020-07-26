#pragma once

#include <stdexcept>

#include "i_complex_network.hpp"

namespace cnn
{
  template <typename T>
  class ComplexNetwork : public IComplexNetwork<T>
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    void SetNetwork2D(typename INetwork2D<T>::Uptr&& network2D) override;
    void SetPerceptron(typename IPerceptron<T>::Uptr&& perceptron) override;

    const INetwork2D<T>& GetNetwork2D() const override;
    INetwork2D<T>& GetNetwork2D() override;

    const IPerceptron<T>& GetPerceptron() const override;
    IPerceptron<T>& GetPerceptron() override;

    void Process() override;

  private:

    typename INetwork2D<T>::Uptr Network2D_;
    typename IPerceptron<T>::Uptr Perceptron_;

  };

  template <typename T>
  void ComplexNetwork<T>::SetNetwork2D(typename INetwork2D<T>::Uptr&& network2D)
  {
    Network2D_ = std::move(network2D);
  }

  template <typename T>
  void ComplexNetwork<T>::SetPerceptron(typename IPerceptron<T>::Uptr&& perceptron)
  {
    Perceptron_ = std::move(perceptron);
  }

  template <typename T>
  const INetwork2D<T>& ComplexNetwork<T>::GetNetwork2D() const
  {
    if (Network2D_ == nullptr)
    {
      throw std::logic_error("cnn::ComplexNetwork::GetNetwork2D() const, Network2D_ == nullptr.");
    }
    return *Network2D_;
  }

  template <typename T>
  INetwork2D<T>& ComplexNetwork<T>::GetNetwork2D()
  {
    if (Network2D_ == nullptr)
    {
      throw std::logic_error("cnn::ComplexNetwork::GetNetwork2D(), Network2D_ == nullptr.");
    }
    return *Network2D_;
  }

  template <typename T>
  const IPerceptron<T>& ComplexNetwork<T>::GetPerceptron() const
  {
    if (Perceptron_ == nullptr)
    {
      throw std::logic_error("cnn::ComplexNetwork::GetPerceptron() const, Perceptron_ == nullptr.");
    }
    return *Perceptron_;
  }

  template <typename T>
  IPerceptron<T>& ComplexNetwork<T>::GetPerceptron()
  {
    if (Perceptron_ == nullptr)
    {
      throw std::logic_error("cnn::ComplexNetwork::GetPerceptron(), Perceptron_ == nullptr.");
    }
    return *Perceptron_;
  }

  template <typename T>
  void ComplexNetwork<T>::Process()
  {
    if (Network2D_ == nullptr)
    {
      throw std::logic_error("cnn::ComplexNetwork::Process(), Network2D_ == nullptr.");
    }
    if (Perceptron_ == nullptr)
    {
      throw std::logic_error("cnn::ComplexNetwork::Process(), Perceptron_ == nullptr.");
    }

    const size_t outputWidth = Network2D_->GetLastLayer().GetOutputWidth();
    const size_t outputHeight = Network2D_->GetLastLayer().GetOutputHeight();

    const size_t m1 = outputWidth * outputHeight;
    if ((m1 / outputWidth) != outputHeight)
    {
      throw std::overflow_error("cnn::ComplexNetwork::Process(), m1 was overflowed.");
    }

    const size_t o = Network2D_->GetLastLayer().GetOutputCount();

    const size_t m2 = m1 * o;
    if ((m2 / m1) != o)
    {
      throw std::logic_error("cnn::ComplexNetwork::Process(), m2 was overflowed.");
    }

    const size_t inputCount = Perceptron_->GetInputCount();
    if (m2 != inputCount)
    {
      throw std::logic_error("cnn::ComplexNetwork::Process(), outputCount != inputCount.");
    }

    Network2D_->Process();

    size_t i{};

    for (size_t o = 0; o < Network2D_->GetLastLayer().GetOutputCount(); ++o)
    {
      for (size_t x = 0; x < Network2D_->GetLastLayer().GetOutputWidth(); ++x)
      {
        for (size_t y = 0; y < Network2D_->GetLastLayer().GetOutputHeight(); ++y)
        {
          const T value = Network2D_->GetLastLayer().GetOutput(o).GetValue(x, y);
          Perceptron_->SetInput(i++, value);
        }
      }
    }

    Perceptron_->Process();
  }
}