#pragma once

#include <memory>
#include <type_traits>

#include "i_map_2d.hpp"
#include "i_filter_2d.hpp"
#include "../common/i_value_generator.hpp"
#include "../common/i_binary_random_generator.hpp"
#include "../common/i_mutagen.hpp"
#include "../common/i_activation_function.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class ILayer2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ILayer2D<T>>;

        virtual size_t GetInputWidth() const = 0;
        virtual size_t GetInputHeight() const = 0;
        virtual size_t GetInputCount() const = 0;
        virtual const IMap2D<T>& GetInput(const size_t index) const = 0;
        virtual IMap2D<T>& GetInput(const size_t index) = 0;

        virtual size_t GetFilterWidth() const = 0;
        virtual size_t GetFilterHeight() const = 0;
        virtual size_t GetFilterCount() const = 0;
        virtual IFilter2D<T>& GetFilter(const size_t index) = 0;
        virtual const IFilter2D<T>& GetFilter(const size_t index) const = 0;

        virtual size_t GetOutputWidth() const = 0;
        virtual size_t GetOutputHeight() const = 0;
        virtual size_t GetOutputCount() const = 0;
        virtual const IMap2D<T>& GetOutput(const size_t index) const = 0;
        virtual IMap2D<T>& GetOutput(const size_t index) = 0;

        virtual void Process() = 0;


        virtual size_t GetOutputValueCount() const = 0;

        // The result must not be nullptr.
        virtual typename ILayer2D<T>::Uptr Clone(const bool cloneState) const = 0;

        virtual void FillWeights(common::IValueGenerator<T>& valueGenerator) = 0;

        virtual void Mutate(common::IMutagen<T>& mutagen) = 0;

        virtual void SetActivationFunctions(const common::IActivationFunction<T>& activationFunction) = 0;

        virtual ~ILayer2D() = default;

      };
    }
  }
}