#pragma once

#include <memory>
#include <type_traits>

#include "map_2d.hpp"
#include "filter_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class IConvolutionHandler2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<IConvolutionHandler2D<T>>;

        virtual size_t GetInputWidth() const = 0;
        virtual size_t GetInputHeight() const = 0;
        virtual size_t GetInputCount() const = 0;

        virtual size_t GetFilterWidth() const = 0;
        virtual size_t GetFilterHeight() const = 0;
        virtual size_t GetFilterCount() const = 0;

        virtual size_t GetOutputWidth() const = 0;
        virtual size_t GetOutputHeight() const = 0;
        virtual size_t GetOutputCount() const = 0;

        virtual IMap2D<T>& GetInput(const size_t index) = 0;
        virtual const IMap2D<T>& GetInput(const size_t index) const = 0;

        virtual IFilter2D<T>& GetFilter(const size_t index) = 0;
        virtual const IFilter2D<T>& GetFilter(const size_t index) const = 0;

        virtual IMap2D<T>& GetOutput(const size_t index) = 0;
        virtual const IMap2D<T>& GetOutput(const size_t index) const = 0;

        virtual void Process() = 0;

        virtual void ClearInputs() = 0;
        virtual void ClearFilters() = 0;
        virtual void ClearOutputs() = 0;

      };
    }
  }
}