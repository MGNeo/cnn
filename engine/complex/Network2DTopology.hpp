#pragma once

#include "../convolution/Network2DTopology.hpp"
#include "../perceptron/NetworkTopology.hpp"
#include "../common/ValueGenerator.hpp"
#include "../common/Mutagen.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      // It is only a container type for parameters, so it doesn't validate contained values.
      // Contained values are checked by a type, which takes this type as parameter.
      // For example, convolution::Network2D validates correctness of convolution::Network2DTopology.
      // It is that, because only consumer knows the rules of the validating.
      class Network2DTopology
      {
      public:

        Network2DTopology(const convolution::Network2DTopology& convolutionTopology = {},
                          const perceptron::NetworkTopology& perceptronTopology = {});

        Network2DTopology(const Network2DTopology& topology) = default;
        
        Network2DTopology(Network2DTopology&& topology) noexcept = default;

        // Exception guarantee: strong for this.
        Network2DTopology& operator=(const Network2DTopology& topology);

        Network2DTopology& operator=(Network2DTopology&& topology) noexcept = default;

        const convolution::Network2DTopology& GetConvolutionTopology() const;

        void SetConvolutionTopology(const convolution::Network2DTopology& convolutionTopology);

        const perceptron::NetworkTopology& GetPerceptronTopology() const;

        void SetPerceptronTopology(const perceptron::NetworkTopology& perceptronTopology);

        // It resets the state to zero.
        void Reset() noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        // It loads full state.
        void Load(std::istream& istream);

      private:

        convolution::Network2DTopology ConvolutionTopology;
        perceptron::NetworkTopology PerceptronTopology;

      };
    }
  }
}
