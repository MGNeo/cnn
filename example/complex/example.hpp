#pragma once

#include "../common/i_example.hpp"

#include "../../engine/convolution/network_2d.hpp"
#include "../../engine/perceptron/network.hpp"
#include "../../engine/complex/network_2d.hpp"

#include "../../engine/complex/lesson_2d.hpp"
#include "../../engine/complex/lesson_2d_library.hpp"

#include "../../engine/complex/genetic_algorithm_2d.hpp"

// Temporary
#include "../../engine/common/binary_random_generator.hpp"

namespace cnn
{
  namespace example
  {
    namespace complex
    {
      template <typename T>
      class Example : public common::IExample<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        void Execute() const override;

      private:

        void Simple() const;
        void Lesson2DLibrary() const;
        void GeneticAlgorithm2D() const;

      };

      template <typename T>
      void Example<T>::Execute() const
      {
        std::cout << __FUNCSIG__ << std::endl;
        {
          Simple();
          Lesson2DLibrary();
          GeneticAlgorithm2D();
        }
        std::cout << std::endl;
      }

      template <typename T>
      void Example<T>::Simple() const
      {
        std::cout << "  " << __FUNCSIG__ << std::endl;
        {
          // Create new 2D convolution subnetwork.
          typename cnn::engine::convolution::Network2D<T>::Uptr subNetwork2D = std::make_unique<cnn::engine::convolution::Network2D<T>>(32, 32, 3, 5, 5, 5);

          // Add few layers to the subnetwork.
          {
            subNetwork2D->PushBack(8, 8, 5);
            subNetwork2D->PushBack(5, 5, 15);
          }

          // Create new perceptron subnetwork.
          // First layer of the subnetwork has 4 neurons.
          const size_t inputCount = subNetwork2D->GetLastLayer().GetOutputValueCount();
          typename cnn::engine::perceptron::Network<T>::Uptr subNetwork = std::make_unique<cnn::engine::perceptron::Network<T>>(inputCount, 4);

          // Add few layers to the subnetwork.
          {
            // Second layer of the subnetwork has 8 neurons.
            subNetwork->PushBack(8);
            // Third layer of the subnetwork has 3 neurons.
            subNetwork->PushBack(3);
          }

          // Create new complex network.
          typename cnn::engine::complex::INetwork2D<T>::Uptr network2D = std::make_unique<cnn::engine::complex::Network2D<T>>(std::move(subNetwork2D), std::move(subNetwork));

          // Set random signals in first layer of the convolution subnetwork.
          {
            // Put some source into the first layer of the convolution subnetwork.
            {
              auto& firstLayer = network2D->GetConvolutionNetwork2D().GetFirstLayer();
              for (size_t i = 0; i < firstLayer.GetInputCount(); ++i)
              {
                auto& input = firstLayer.GetInput(i);
                for (size_t x = 0; x < firstLayer.GetInputWidth(); ++x)
                {
                  for (size_t y = 0; y < firstLayer.GetInputHeight(); ++y)
                  {
                    // Some value for example.
                    const float value = static_cast<float>(rand()) / RAND_MAX;// ???
                    input.SetValue(x, y, value);
                  }
                }
              }
            }
          }

          // Pass signal through the complex network.
          {
            network2D->Process();

          }

          // Take the result from the last layer of the perceptron subnetwork.
          {
            const auto& outputLayer = network2D->GetPerceptronNetwork().GetLastLayer();
            // ...
          }

          // Temporary.
          {
            auto binaryRandomGenerator = std::make_unique<engine::common::BinaryRandomGenerator>(0);
            //network2D->CrossFrom(*network2D, *network2D, *binaryRandomGenerator);
          }
        }
      }

      template <typename T>
      void Example<T>::Lesson2DLibrary() const
      {
        std::cout << "  " << __FUNCSIG__ << std::endl;
        {
          // Create new lesson library.
          typename cnn::engine::complex::ILesson2DLibrary<T>::Uptr lesson2DLibrary = std::make_unique<cnn::engine::complex::Lesson2DLibrary<T>>(32, 32, 3, 5);

          // Fill the library with empty lessons.
          {
            lesson2DLibrary->PushBack();
            lesson2DLibrary->PushBack();
            lesson2DLibrary->PushBack();
          }

          // Fill the last lesson in the library.
          {
            auto& lesson2D = lesson2DLibrary->GetLastLesson();
            lesson2D.GetInput(0).SetValue(0, 0, 1.f);
            // TODO: Add more detailed examples.
            // ...
          }

          // TODO:
          // ...
        }
        std::cout << std::endl;
      }

      template <typename T>
      void Example<T>::GeneticAlgorithm2D() const
      {
        std::cout << "  " << __FUNCSIG__ << std::endl;
        {
          auto geneticAlgorithm2D = std::make_unique<engine::complex::GeneticAlgorithm2D<T>>();
        }
        std::cout << std::endl;
      }
    }
  }
}