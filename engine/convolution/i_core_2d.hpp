#pragma once

#include <memory>
#include <type_traits>

#include "../common/i_value_generator.hpp"
#include "../common/i_binary_random_generator.hpp"
#include "../common/i_neuron.hpp"
#include "../common/i_mutagen.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class ICore2D
      {
        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ICore2D<T>>;

        virtual size_t GetWidth() const = 0;
        virtual size_t GetHeight() const = 0;

        virtual T GetInput(const size_t x, const size_t y) const = 0;
        virtual void SetInput(const size_t x, const size_t y, const T value) = 0;

        virtual T GetWeight(const size_t x, const size_t y) const = 0;
        virtual void SetWeight(const size_t x, const size_t y, const T value) = 0;

        virtual void Process() = 0;

        virtual T GetOutput() const = 0;

        virtual void ClearInputs() = 0;
        virtual void ClearWeights() = 0;
        virtual void ClearOutput() = 0;

        virtual ~ICore2D() = default;

        virtual typename ICore2D<T>::Uptr Clone(const bool cloneState) const = 0;

        virtual void FillWeights(common::IValueGenerator<T>& valueGenerator) = 0;

        virtual void CrossFrom(const ICore2D<T>& source1,
                               const ICore2D<T>& source2,
                               common::IBinaryRandomGenerator& binaryRandomGenerator) = 0;

        virtual void Mutate(common::IMutagen<T>& mutagen) = 0;

      };
    }
  }
}