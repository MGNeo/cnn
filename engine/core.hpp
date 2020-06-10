#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <stdexcept>

#include "standard_activator.hpp"

namespace cnn
{
  namespace engine
  {
    template <typename T, size_t C, typename F = StandardActivator<T>>
    class Core
    {

      static_assert(std::is_arithmetic <T>::value);
      static_assert(C > 0);

    public:

      Core();

      T GetInput(const size_t index) const;
      void SetInput(const size_t index, const T value);

      T GetWeight(const size_t index) const;
      void SetWeight(const size_t index, const T value);

      T GetOutput() const;

      void GenerateOutput();

      size_t GetInputCount() const;

    private:

      std::array<T, C> Inputs;
      std::array<T, C> Weights;
      T Output;

    };

    template <typename T, size_t C, typename F>
    Core<T, C, F>::Core()
      :
      Inputs{},
      Weights{},
      Output{}
    {
    }

    template <typename T, size_t C, typename F>
    T Core<T, C, F>::GetInput(const size_t index) const
    {
      if (index >= C)
      {
        throw std::range_error("cnn::engine::Core::GetInput(), index >= C.");
      }
      return Inputs[index];
    }

    template <typename T, size_t C, typename F>
    void Core<T, C, F>::SetInput(const size_t index, const T value)
    {
      if (index >= C)
      {
        throw std::range_error("cnn::engine::Core::SetInput(), index >= C.");
      }
      Inputs[index] = value;
    }

    template <typename T, size_t C, typename F>
    T Core<T, C, F>::GetWeight(const size_t index) const
    {
      if (index >= C)
      {
        throw std::range_error("cnn::engine::Core::GetWeight(), index >= C.");
      }
      return Weights[index];
    }

    template <typename T, size_t C, typename F>
    void Core<T, C, F>::SetWeight(const size_t index, const T value)
    {
      if (index >= C)
      {
        throw std::range_error("cnn::engine::Core::SetWeight(), index >= C.");
      }
      Weights[index] = value;
    }

    template <typename T, size_t C, typename F>
    T Core<T, C, F>::GetOutput() const
    {
      return Output;
    }

    template <typename T, size_t C, typename F>
    void Core<T, C, F>::GenerateOutput()
    {
      T result{};
      for (size_t i = 0; i < C; ++i)
      {
        result += F::Handle(Inputs[i] * Weights[i]);
      }
      return result;
    }

    template<typename T, size_t C, typename F>
    size_t Core<T, C, F>::GetInputCount() const
    {
      return C;
    }
  }
}
