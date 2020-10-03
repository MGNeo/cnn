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

        typename typename INetwork2D<T>::Uptr Clone(const bool cloneState) const override;

        void FillWeights(common::IValueGenerator<T>& valueGenerator) override;

        void CrossFrom(const INetwork2D<T>& source1,
                       const INetwork2D<T>& source2,
                       common::IBinaryRandomGenerator& binaryRandomGenerator) override;

        void Mutate(common::IMutagen<T>& mutagen) override;

      private:

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
        if (convolutionNetwork2D->GetLastLayer().GetOutputValueCount() != perceptronNetwork->GetFirstLayer().GetInputSize())
        {
          throw std::invalid_argument("cnn::engine::complex::Network2D::Network2D(), convolutionNetwork2D->GetLastLayer().GetOutputValueCount() != perceptronNetwork->GetFirstLayer().GetInputSize().");
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
              for (size_t y = 0; y < cLastLayer.GetOutputHeight(); ++y)
              {
                const T value = cOutput.GetValue(x, y);
                pInput.SetValue(i++, value);
              }
            }
          }
        }
        PerceptronNetwork->Process();
      }

      template <typename T>
      typename INetwork2D<T>::Uptr Network2D<T>::Clone(const bool cloneState) const
      {
        auto convolutionNetwork2D = ConvolutionNetwork2D->Clone(cloneState);
        auto perceptronNetwork = PerceptronNetwork->Clone(cloneState);
        auto complexNetwork = std::make_unique<Network2D<T>>(std::move(convolutionNetwork2D),
                                                             std::move(perceptronNetwork));
        return complexNetwork;
      }

      template <typename T>
      void Network2D<T>::FillWeights(common::IValueGenerator<T>& valueGenerator)
      {
        ConvolutionNetwork2D->FillWeights(valueGenerator);
        PerceptronNetwork->FillWeights(valueGenerator);
      }

      template <typename T>
      void Network2D<T>::CrossFrom(const INetwork2D<T>& source1,
                                   const INetwork2D<T>& source2,
                                   common::IBinaryRandomGenerator& binaryRandomGenerator)
      {
        ConvolutionNetwork2D->CrossFrom(source1.GetConvolutionNetwork2D(), source2.GetConvolutionNetwork2D(), binaryRandomGenerator);
        PerceptronNetwork->CrossFrom(source1.GetPerceptronNetwork(), source2.GetPerceptronNetwork(), binaryRandomGenerator);
      }

      template <typename T>
      void Network2D<T>::Mutate(common::IMutagen<T>& mutagen)
      {
        ConvolutionNetwork2D->Mutate(mutagen);
        PerceptronNetwork->Mutate(mutagen);
      }
    }
  }
}