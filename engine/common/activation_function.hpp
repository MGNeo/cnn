#pragma once

#include "i_activation_function.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class ActivationFunction : public IActivationFunction<T>
      {
      public:

        using Uptr = std::unique_ptr<ActivationFunction<T>>;

        T Use(const T value) const override;

        // The result must not be nullptr.
        typename IActivationFunction<T>::Uptr Clone() const override;

      };

      template <typename T>
      T ActivationFunction<T>::Use(const T value) const
      {
        if (value > 0)
        {
          return sqrt(value);
        } else {
          return sqrt(-value);
        }
      }

      // The result must not be nullptr.
      template <typename T>
      typename IActivationFunction<T>::Uptr ActivationFunction<T>::Clone() const
      {
        return std::make_unique<ActivationFunction<T>>();
      }
    }
  }
}