#include "Network2DTopology.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      Network2DTopology::Network2DTopology(const convolution::Network2DTopology& convolutionTopology,
                                           const perceptron::NetworkTopology& perceptronTopology)
        :
        ConvolutionTopology{ convolutionTopology },
        PerceptronTopology{ perceptronTopology }
      {
      }

      Network2DTopology& Network2DTopology::operator=(const Network2DTopology& topology)
      {
        if (this != &topology)
        {
          Network2DTopology tmpTopology{ topology };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpTopology);
        }
        return *this;
      }

      convolution::Network2DTopology Network2DTopology::GetConvolutionTopology() const
      {
        return ConvolutionTopology;
      }

      void Network2DTopology::SetConvolutionTopology(const convolution::Network2DTopology& convolutionTopology)
      {
        ConvolutionTopology = convolutionTopology;
      }

      perceptron::NetworkTopology Network2DTopology::GetPerceptronTopology() const
      {
        return PerceptronTopology;
      }

      void Network2DTopology::SetPerceptronTopology(const perceptron::NetworkTopology& perceptronTopology)
      {
        PerceptronTopology = perceptronTopology;
      }

      void Network2DTopology::Reset() noexcept
      {
        ConvolutionTopology.Reset();
        PerceptronTopology.Reset();
      }

      void Network2DTopology::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::complex::Network2DTopology::Save(), ostream.good() == false.");
        }
        ConvolutionTopology.Save(ostream);
        PerceptronTopology.Save(ostream);
        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::complex::Network2DTopology::Save(), ostream.good() == false.");
        }
      }

      void Network2DTopology::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::complex::Network2DTopology::Load(), istream.good() == false.");
        }
        
        decltype(ConvolutionTopology) convolutionTopology;
        decltype(PerceptronTopology) perceptronTopology;

        convolutionTopology.Load(istream);
        perceptronTopology.Load(istream);

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::complex::Network2DTopology::Load(), istream.good() == false.");
        }

        ConvolutionTopology = std::move(convolutionTopology);
        PerceptronTopology = std::move(perceptronTopology);
      }
    }
  }
}