#pragma once

#include <memory>
#include <type_traits>

#include "i_value_generator.hpp"
#include "i_binary_random_generator.hpp"
#include "i_mutagen.hpp"
#include "i_activation_function.hpp"

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
        virtual void ClearWeights() = 0;
        virtual void ClearOutput() = 0;

        virtual ~INeuron() = default;

        // The result must not be nullptr.
        virtual typename INeuron<T>::Uptr Clone(const bool cloneState) const = 0;

        virtual void FillWeights(IValueGenerator<T>& valueGenerator) = 0;

        virtual void CrossFrom(const INeuron<T>& source1,
                               const INeuron<T>& source2,
                               IBinaryRandomGenerator& binaryRandomGenerator) = 0;

        virtual void Mutate(IMutagen<T>& mutagen) = 0;

        virtual const IActivationFunction<T>& GetActivationFunction() const = 0;
        virtual void SetActivationFunction(const IActivationFunction<T>& activationFunction) = 0;

      };
    }
  }
}