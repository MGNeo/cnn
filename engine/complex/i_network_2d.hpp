#pragma once

#include <type_traits>
#include <memory>
#include <string>

#include "../convolution/i_network_2d.hpp"
#include "../perceptron/i_network.hpp"
#include "../common/i_value_generator.hpp"
#include "../common/i_binary_random_generator.hpp"
#include "../common/i_activation_function.hpp"

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

        // The result must not be nullptr.
        virtual typename INetwork2D<T>::Uptr Clone(const bool cloneState) const = 0;

        virtual void FillWeights(common::IValueGenerator<T>& valueGenerator) = 0;

        virtual void Mutate(common::IMutagen<T>& mutagen) = 0;

        virtual void SetActivationFunctions(const common::IActivationFunction<T>& activationFunction) = 0;

        // Save weights values to file.
        virtual void Save(const std::string& fileName) const = 0;

        // Load weight values from file.
        // Static types, dynamic types and topology of all subitems between
        // this network and the network loaded from the file must be equal.
        // Otherwise the behavior is undefined.

        // Of course, we can implement full saving and loading of the network,
        // but it need much more time, so we hasn't done this.
        virtual void Load(const std::string& fileName) = 0;

        virtual ~INetwork2D() = default;

      };
    }
  }
}