#pragma once

#include <memory>
#include <type_traits>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class INeuron
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<INeuron<T>>;

        virtual size_t GetInputCount() const = 0;

        virtual T GetInput(const size_t index) const = 0;
        virtual void SetInput(const size_t index, const T value) = 0;

        virtual T GetWeight(const size_t index) const = 0;
        virtual void SetWeight(const size_t index, const T value) = 0;

        virtual void Process() = 0;

        virtual T GetOutput() const = 0;

        virtual void ClearInputs() = 0;
        virtual void ClearWeight() = 0;
        virtual void ClearOutput() = 0;

        virtual ~INeuron() = default;

        virtual typename INeuron<T>::Uptr Clone(const bool cloneState) const = 0;

      };
    }
  }
}