#pragma once

#include <memory>
#include <type_traits>

#include "i_core_2d.hpp"
#include "../common/i_value_generator.hpp"
#include "../common/i_binary_random_generator.hpp"
#include "../common/i_mutagen.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class IFilter2D
      {
        
        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<IFilter2D<T>>;

        virtual size_t GetWidth() const = 0;
        virtual size_t GetHeight() const = 0;

        virtual size_t GetCoreCount() const = 0;

        virtual ICore2D<T>& GetCore(const size_t index) = 0;
        virtual const ICore2D<T>& GetCore(const size_t index) const = 0;

        virtual void Clear() = 0;
        virtual void ClearInputs() = 0;
        virtual void ClearWeight() = 0;
        virtual void ClearOutput() = 0;

        virtual ~IFilter2D() = default;

        virtual typename IFilter2D<T>::Uptr Clone(const bool cloneState) const = 0;

        virtual void CrossFrom(const IFilter2D<T>& source1,
                               const IFilter2D<T>& source2,
                               common::IBinaryRandomGenerator& binaryRandomGenerator) = 0;

        virtual void FillWeights(common::IValueGenerator<T>& valueGenerator) = 0;

        virtual void Mutate(common::IMutagen<T>& mutagen) = 0;

      };
    }
  }
}