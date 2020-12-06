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

        // It resets the state to zero including the topology.
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
