#pragma once

#include <filesystem>

#include "i_factory.hpp"

#include "3rd/SFML-2.5.1/include/SFML/Graphics.hpp"

#include "engine/complex/lesson_2d_library.hpp"
#include "engine/complex/network_2d.hpp"
#include "engine/convolution/network_2d.hpp"
#include "engine/perceptron/network.hpp"
#include "engine/complex/genetic_algorithm_2d.hpp"
#include "engine/common/value_generator.hpp"

namespace cnn
{
  namespace examples
  {
    namespace complex_learning
    {
      template <typename T>
      class Factory : public IFactory<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Factory();

        // The result must not be nullptr.
        typename engine::complex::ILesson2DLibrary<T>::Uptr Library() const override;
        // The result must not be nullptr.
        typename engine::complex::INetwork2D<T>::Uptr Network() const override;
        // The result must not be nullptr.
        typename engine::complex::IGeneticAlgorithm2D<T>::Uptr Algorithm() const override;

      private:

        size_t InputWidth;
        size_t InputHeight;
        size_t InputCount;
        size_t OutputCount;

        void LoadNumbers(engine::complex::ILesson2DLibrary<T>& library) const;
        void LoadNumber(engine::complex::ILesson2DLibrary<T>& library, const size_t number) const;

      };

      template <typename T>
      Factory<T>::Factory()
        :
        InputWidth{ 32 },
        InputHeight{ 32 },
        InputCount{ 1 },
        OutputCount{ 10 }
      {
      }

      // The result must not be nullptr.
      template <typename T>
      typename engine::complex::ILesson2DLibrary<T>::Uptr Factory<T>::Library() const
      {
        
        auto library = std::make_unique<engine::complex::Lesson2DLibrary<T>>(InputWidth,
                                                                             InputHeight,
                                                                             InputCount,
                                                                             OutputCount);

        LoadNumbers(*library);

        return std::move(library);
      }

      template <typename T>
      void Factory<T>::LoadNumbers(engine::complex::ILesson2DLibrary<T>& library) const
      {
        for (size_t number = 0; number < OutputCount; ++number)
        {
          LoadNumber(library, number);
        }
      }

      template <typename T>
      void Factory<T>::LoadNumber(engine::complex::ILesson2DLibrary<T>& library, const size_t number) const
      {
        if (number >= OutputCount)
        {
          throw std::invalid_argument("cnn::examplex::Factory::LoadNumber(), number >= OutputCount.");
        }

        const std::string path = "../../data/numbers/" + std::to_string(number) + "/";

        std::filesystem::directory_iterator di{ path };

        for (auto de : di)
        {
          if (de.is_regular_file() && (de.path().extension().string() == ".bmp"))
          {
            sf::Image image;
            if (image.loadFromFile(de.path().string()))
            {
              if ((image.getSize().x == InputWidth) && (image.getSize().y == InputHeight))
              {
                library.PushBack();
                auto& lesson = library.GetLastLesson();
                for (size_t x = 0; x < InputWidth; ++x)
                {
                  for (size_t y = 0; y < InputHeight; ++y)
                  {
                    T value{};

                    if (image.getPixel(static_cast<unsigned int>(x), static_cast<unsigned int>(y)) == sf::Color::Black)
                    {
                      value = 1;
                    }

                    lesson.GetInput(0).SetValue(x, y, value);
                  }
                }
                lesson.GetOutput().SetValue(number, 1);
              }
            }
          }
        }
      }

      // The result must not be nullptr.
      template <typename T>
      typename engine::complex::INetwork2D<T>::Uptr Factory<T>::Network() const
      {
        // Prepare convolution network.
        auto convolutionNetwork = std::make_unique<engine::convolution::Network2D<T>>(InputWidth, InputHeight, InputCount, 3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(2, 2, 10);

        // Prepare perceptron network.
        auto perceptronNetwork = std::make_unique<engine::perceptron::Network<T>>(convolutionNetwork->GetLastLayer().GetOutputValueCount(), 15);
        perceptronNetwork->PushBack(OutputCount);

        // Prepare complex network.
        auto complexNetwork = std::make_unique<engine::complex::Network2D<T>>(std::move(convolutionNetwork), std::move(perceptronNetwork));

        // Fill the complex network with noise.
        auto valueGenerator = std::make_unique<engine::common::ValueGenerator<T>>();
        valueGenerator->SetMinValue(-5);
        valueGenerator->SetMaxValue(+5);
        complexNetwork->FillWeights(*valueGenerator);

        return std::move(complexNetwork);
      }

      // The result must not be nullptr.
      template <typename T>
      typename engine::complex::IGeneticAlgorithm2D<T>::Uptr Factory<T>::Algorithm() const
      {
        auto algorithm = std::make_unique<engine::complex::GeneticAlgorithm2D<T>>();

        algorithm->SetIterationCount(10);
        algorithm->SetThreadCount(8);

        // Configure the mutagen which mutates the weights of the networks.
        auto mutagen = algorithm->GetMutagen().Clone();
        mutagen->SetMinResult(static_cast<T>(-5L));
        mutagen->SetMaxResult(static_cast<T>(+5L));
        mutagen->SetVariabilityForce(static_cast<T>(0.1L));
        algorithm->SetMutagen(*mutagen);

        return std::move(algorithm);
      }
    }
  }
}

