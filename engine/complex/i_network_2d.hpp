#pragma once

#include <type_traits>
#include <memory>

#include "../convolution/i_network_2d.hpp"
#include "../perceptron/i_network.hpp"
#include "../common/i_value_generator.hpp"
#include "../common/i_binary_random_generator.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class INetwork2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<INetwork2D<T>>;

        virtual const convolution::INetwork2D<T>& GetConvolutionNetwork2D() const = 0;
        virtual convolution::INetwork2D<T>& GetConvolutionNetwork2D() = 0;

        virtual const perceptron::INetwork<T>& GetPerceptronNetwork() const = 0;
        virtual perceptron::INetwork<T>& GetPerceptronNetwork() = 0;

        virtual void Process() = 0;

        virtual typename INetwork2D<T>::Uptr Clone(const bool cloneState) const = 0;

        virtual void FillWeights(common::IValueGenerator<T>& valueGenerator) = 0;

        virtual void CrossFrom(const INetwork2D<T>& source1,
                               const INetwork2D<T>& source2,
                               common::IBinaryRandomGenerator& binaryRandomGenerator) = 0;

        virtual ~INetwork2D() = default;

      };
    }
  }
}