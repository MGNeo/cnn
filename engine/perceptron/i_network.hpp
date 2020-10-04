#pragma once

#include <type_traits>
#include <memory>

#include "i_layer.hpp"
#include "../common/i_value_generator.hpp"
#include "../common/i_binary_random_generator.hpp"
#include "../common/i_mutagen.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      template <typename T>
      class INetwork
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<INetwork<T>>;

        virtual size_t GetLayerCount() const = 0;

        virtual const ILayer<T>& GetLayer(const size_t index) const = 0;
        virtual ILayer<T>& GetLayer(const size_t index) = 0;

        virtual const ILayer<T>& GetLastLayer() const = 0;
        virtual ILayer<T>& GetLastLayer() = 0;

        virtual const ILayer<T>& GetFirstLayer() const = 0;
        virtual ILayer<T>& GetFirstLayer() = 0;

        virtual void Process() = 0;

        // TODO: Clear() family methods.

        virtual typename INetwork<T>::Uptr Clone(const bool cloneState) const = 0;

        virtual void FillWeights(common::IValueGenerator<T>& valueGenerator) = 0;

        virtual void CrossFrom(const INetwork<T>& source1,
                               const INetwork<T>& source2,
                               common::IBinaryRandomGenerator& binaryRandomGenerator) = 0;

        virtual void Mutate(common::IMutagen<T>& mutagen) = 0;

        virtual void SetActivationFunctions(const common::IActivationFunction<T>& activationFunction) = 0;

        virtual ~INetwork() = default;

      };
    }
  }
}