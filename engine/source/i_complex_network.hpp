#pragma once

#include <memory>

#include "i_network_2d.hpp"
#include "i_perceptron.hpp"

namespace cnn
{
  template <typename T>
  class IComplexNetwork
  {

    static_assert(std::is_floating_point<T>::value);

  public:

    using Uptr = std::unique_ptr<IComplexNetwork<T>>;

    virtual void SetNetwork2D(typename INetwork2D<T>::Uptr&& network2D) = 0;
    virtual void SetPerceptron(typename IPerceptron<T>::Uptr&& perceptron) = 0;

    virtual const INetwork2D<T>& GetNetwork2D() const = 0;
    virtual INetwork2D<T>& GetNetwork2D() = 0;

    virtual const IPerceptron<T>& GetPerceptron() const = 0;
    virtual IPerceptron<T>& GetPerceptron() = 0;

    virtual void Process() = 0;

    virtual ~IComplexNetwork() {}

  };
}