#pragma once

#include <memory>
#include <type_traits>

#include "i_layer_2d.hpp"
#include "../common/i_value_generator.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class INetwork2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<INetwork2D<T>>;
        
        virtual size_t GetLayerCount() const = 0;

        virtual const ILayer2D<T>& GetLayer(const size_t index) const = 0;
        virtual ILayer2D<T>& GetLayer(const size_t index) = 0;

        virtual const ILayer2D<T>& GetLastLayer() const = 0;
        virtual ILayer2D<T>& GetLastLayer() = 0;

        virtual const ILayer2D<T>& GetFirstLayer() const = 0;
        virtual ILayer2D<T>& GetFirstLayer() = 0;

        virtual void Process() = 0;

        virtual typename INetwork2D<T>::Uptr Clone(const bool cloneState) const = 0;

        virtual void FillWeights(common::IValueGenerator<T>& valueGenerator) = 0;

        //virtual void CrossFrom(const INetwork2D<T>& source1,
        //                       const INetwork2D<T>& source2) = 0;

        virtual ~INetwork2D() = default;

      };
    }
  }
}