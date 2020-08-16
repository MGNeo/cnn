#pragma once

#include "i_network_2d.hpp"

#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class Network2D : public INetwork2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<Network2D<T>>;

        Network2D(typename convolution::INetwork2D<T>::Uptr&& convolutionNetwork2D,
                  typename perceptron::INetwork<T>::Uptr&& perceptronNetwork);

        const convolution::INetwork2D<T>& GetConvolutionNetwork2D() const override;
        convolution::INetwork2D<T>& GetConvolutionNetwork2D() override;

        const perceptron::INetwork<T>& GetPerceptronNetwork() const override;
        perceptron::INetwork<T>& GetPerceptronNetwork() override;

        void Process() override;

      public:

        typename convolution::INetwork2D<T>::Uptr ConvolutionNetwork2D;
        typename perceptron::INetwork<T>::Uptr PerceptronNetwork;

      };

      template <typename T>
      Network2D<T>::Network2D(typename convolution::INetwork2D<T>::Uptr&& convolutionNetwork2D,
                              typename perceptron::INetwork<T>::Uptr&& perceptronNetwork)
      {
        if (convolutionNetwork2D == nullptr)
        {
          throw std::invalid_argument("cnn::engine::complex::Network2D::Network2D(), convolutionNetwork2D == nullptr.");
        }
        if (perceptronNetwork == nullptr)
        {
          throw std::invalid_argument("cnn::engine::complex::Network2D::Network2D(), perceptronNetwork == nullptr.");
        }

        const size_t cOutputWidth = convolutionNetwork2D->GetLastLayer().GetOutputWidth();
        const size_t cOutputHeight = convolutionNetwork2D->GetLastLayer().GetOutputHeight();
        const size_t cOutputCount = convolutionNetwork2D->GetLastLayer().GetOutputCount();

        const size_t m1 = cOutputWidth * cOutputHeight;
        if ((m1 / cOutputWidth) != cOutputHeight)
        {
          throw std::overflow_error("cnn::engine::complex::Network2D::Netowrk2D(), m1 was overflowed.");
        }

        const size_t cTotalOutputCount = m1 * cOutputCount;
        if ((cTotalOutputCount / m1) != cOutputCount)
        {
          throw std::overflow_error("cnn::engine::complex::Network2D::Network2D(), cTotalOutputCount was overflowed.");
        }

        if (cTotalOutputCount != perceptronNetwork->GetLastLayer().GetInputSize())
        {
          throw std::logic_error("cnn::engine::complex::Network2D::Network2D(), cTotalOutputCount != perceptronNetwork->GetLastLayer().GetInputSize().");
        }

        ConvolutionNetwork2D = std::move(convolutionNetwork2D);
        PerceptronNetwork = std::move(perceptronNetwork);
      }

      template <typename T>
      const convolution::INetwork2D<T>& Network2D<T>::GetConvolutionNetwork2D() const
      {
        return *(ConvolutionNetwork2D);
      }

      template <typename T>
      convolution::INetwork2D<T>& Network2D<T>::GetConvolutionNetwork2D()
      {
        return *(ConvolutionNetwork2D);
      }

      template <typename T>
      const perceptron::INetwork<T>& Network2D<T>::GetPerceptronNetwork() const
      {
        return *(PerceptronNetwork);
      }

      template <typename T>
      perceptron::INetwork<T>& Network2D<T>::GetPerceptronNetwork()
      {
        return *(PerceptronNetwork);
      }

      template <typename T>
      void Network2D<T>::Process()
      {
        // TODO: Think about rollback when exception is thrown.
        ConvolutionNetwork2D->Process();
        {
          auto& pFirstLayer = PerceptronNetwork->GetLayer(0);// TODO: Create special method GetFirstLayer().
          auto& pInput = pFirstLayer.GetInput();
          const auto& cLastLayer = ConvolutionNetwork2D->GetLastLayer();
          for (size_t o = 0; o < cLastLayer.GetOutputCount(); ++o)
          {
            const auto& cOutput = cLastLayer.GetOutput(o);
            size_t i{};
            for (size_t x = 0; x < cLastLayer.GetOutputWidth(); ++x)
            {
              for (size_t y = 0; y < cLastLayer.GetOutputHeight; ++y)
              {
                const T value = cOutput.GetValue(x, y);
                pInput.SetValue(i++, value);
              }
            }
          }
        }
        PerceptronNetwork->Process();
      }
    }
  }
}