#pragma once

#include "Lesson2DLibrary.hpp"
#include "Network2D.hpp"

#include <thread>

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class GeneticTest2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        GeneticTest2D(const Lesson2DLibrary<T>& lessonLibrary,
                      const Network2D<T>& network,
                      const size_t threadCount = 0);

        T GetTotalError() const noexcept;

      private:
        
        const T TotalError;

        static T TestThread(const Lesson2DLibrary<T>& lessonLibrary,
                            const Network2D<T>& network,
                            const size_t threadCount,
                            const size_t threadId);

      };

      template <typename T>
      GeneticTest2D<T>::GeneticTest2D(const Lesson2DLibrary<T>& lessonLibrary,
                                      const Network2D<T>& network,
                                      const size_t threadCount)
      {
        const size_t count = threadCount ? threadCount : std::thread::hardware_concurrency();

        // Run threads.

        // Wait the threads.
      }

      template <typename T>
      T GeneticTest2D<T>::GetTotalError() const noexcept
      {
        return TotalError;
      }

      template <typename T>
      T GeneticTest2D<T>::TestThread(const Lesson2DLibrary<T>& lessonLibrary,
                                     const Network2D<T>& network,
                                     const size_t threadCount,
                                     const size_t threadId)
      {
        // If we use "auto", code analiser of Visual Studio dies.
        Network2D<T> copiedNetwork{ network };
        convolution::Network2DProtectingReference<T> convolutionNetwork = copiedNetwork.GetConvolutionNetwork();
        perceptron::NetworkProtectingReference<T> perceptronNetwork = copiedNetwork.GetPerceptronNetwork();
        for (size_t lessonId = threadId; lessonId < lessonLibrary.GetLessonCount(); lessonId += threadCount)
        {
          const complex::Lesson2D<T>& lesson = lessonLibrary.GetLesson(lessonId);
          convolution::Layer2DProtectingReference<T> firstLayer = convolutionNetwork.GetLastLayer(0);
          // Convolution.
          {
            convolutionNetwork.Clear();
            for (size_t inputIndex = 0; inputIndex < lesson.GetTopology().GetInputCount(); ++inputIndex)
            {
              convolution::Map2DProtectingReference<T> input = firstLayer.GetInput(inputIndex);
              for (size_t x = 0; x < lesson.GetSize().GetWidth(); ++x)
              {
                for (size_t y = 0; y < lesson.GetSize().GetHeight(); ++y)
                {
                  input.FillFrom(lesson.GetInput(inputIndex));
                }
              }
            }
            convolutionNetwork.GenerateOutput();
          }
          // Perceptron.
          {
            convolution::Layer2DProtectingReference<T> lastConvolutionLayer = convolutionNetwork.GetLastLayer();
            perceptron::LayerProtectingReference<T> firstPerceptronLayer = perceptronNetwork.GetFirstLayer();
            common::MapProtectingReference<T> firstPerceptronLayerInput = firstPerceptronLayer.GetInput();
            perceptronNetwork.Clear();
            size_t index{};
            for (size_t outputIndex = 0; outputIndex < lastConvolutionLayer.GetTopology().GetOutputCount(); ++outputIndex)
            {
              convolution::Map2DProtectingReference<T> convolutionOutput = lastConvolutionLayer.GetOutput(outputIndex);
              for (size_t x = 0; x < convolutionOutput.GetSize().GetWidth(); ++x)
              {
                for (size_t y = 0; y < convolutionOutput.GetSize().GetHeight(); ++y)
                {
                  const auto value = convolutionOutput.GetValue(x, y);
                  firstPerceptronLayerInput.SetValue(index++, value);
                }
              }
            }
            perceptronNetwork.GenerateOutput();
          }
          // Total error.
          {
            // TODO: Add GetLastLayer() to perceptron::Network.
            //perceptron::LayerProtectingReference<T> lastPerceptronLayer = perceptronNetwork.GetLastLayer();
          }
        }
      }
    }
  }
}