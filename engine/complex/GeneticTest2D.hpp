#pragma once

#include "Lesson2DLibrary.hpp"
#include "Network2D.hpp"
#include "GroupErrorFlag.hpp"

#include <thread>
#include <utility>
#include <future>

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
        
        T TotalError;

        static T TestThread(const Lesson2DLibrary<T>& lessonLibrary,
                            const Network2D<T>& network,
                            const size_t threadCount,
                            const size_t threadId,
                            GroupErrorFlag& groupErrorFlag);

        void CheckTopologies(const Lesson2DLibrary<T>& lessonLibrary,
                             const Network2D<T>& network) const;

      };

      template <typename T>
      GeneticTest2D<T>::GeneticTest2D(const Lesson2DLibrary<T>& lessonLibrary,
                                      const Network2D<T>& network,
                                      const size_t threadCount)
      {
        CheckTopologies(lessonLibrary, network)

        const size_t threafCount_ = threadCount ? threadCount : std::thread::hardware_concurrency();

        // Run threads.
        GroupErrorFlag groupErrorFlag;
        std::list<std::future<T>> futures;
        for (size_t threadId = 0; threadId < threafCount_; ++threadId)
        {
          std::future<T> future = std::async(std::launch::async,
                                             TestThread,
                                             lessonLibrary,
                                             network,
                                             threadCount_,
                                             threadId,
                                             groupErrorFlag);

          futures.push_back(std::move(future));
        }

        // If several errors are occurred, only the once exception is thrown (the first).
        T totalError{};
        for (auto& future : futures)
        {
          totalError += future.get();
        }

        return totalError;
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
                                     const size_t threadId,
                                     GroupErrorFlag& groupErrorFlag)
      {
        try
        {
          // If we use "auto", then code analiser of Visual Studio dies.
          Network2D<T> copiedNetwork{ network };
          convolution::Network2DProtectingReference<T> convolutionNetwork = copiedNetwork.GetConvolutionNetwork();
          perceptron::NetworkProtectingReference<T> perceptronNetwork = copiedNetwork.GetPerceptronNetwork();
          for (size_t lessonId = threadId;
               (lessonId < lessonLibrary.GetLessonCount()) && (groupErrorFlag.IsError() == false);
               lessonId += threadCount)
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
                    const T value = convolutionOutput.GetValue(x, y);
                    firstPerceptronLayerInput.SetValue(index++, value);
                  }
                }
              }
              perceptronNetwork.GenerateOutput();
            }
            // Total error.
            {
              T totalError{};
              const common::Map<T>& perceptronOutput = perceptronNetwork.GetLastLayer().GetOutput();
              const common::Map<T>& lessonOutput = lesson.GetOutput();
              for (size_t o = 0; o < perceptronOutput.GetValueCount(); ++o)
              {
                const T perceptronOutputValue = perceptronOutput.GetValue(o);
                const T lessonOutputValue = lessonOutput.GetValue(o);
                totalError += abs(perceptronOutput - lessonOutput);
              }
            }
          }
        }
        catch (...)
        {
          groupErrorFlag.SetUp();
          throw;
        }
        return totalError;
      }

      template <typename T>
      void GeneticTest2D<T>::CheckTopologies(const Lesson2DLibrary<T>& lessonLibrary,
                                             const Network2D<T>& network) const
      {
        if (lessonLibrary.GetLessonCount() == 0)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticTest2D::CheckTopologies(), lessonLibrary.GetLessonCount() == 0.");
        }
        if (lessonLibrary.GetLesson(0).GetTopology().GetInputSize() != network.GetConvolutionNetwork().GetTopology().GetFirstLayerTopology().GetInputSize())
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticTest2D::CheckTopologies(), lessonLibrary.GetLesson(0).GetTopology().GetInputSize() != network.GetConvolutionNetwork().GetTopology().GetFirstLayerTopology().GetInputSize().");
        }
        if (lessonLibrary.GetLesson(0).GetTopology().GetInputCount() != network.GetConvolutionNetwork().GetTopology().GetFirstLayerTopology().GetInputCount())
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticTest2D::CheckTopologies(), lessonLibrary.GetLesson(0).GetTopology().GetInputCount() != network.GetConvolutionNetwork().GetTopology().GetFirstLayerTopology().GetInputCount().");
        }
        if (lessonLibrary.GetLesson(0).GetTopology().GetOutputCount() != network.GetPerceptronNetwork().GetTopology().GetFirstLayerTopology().GetNeuronCount())
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticTest2D::CheckTopologies(), lessonLibrary.GetLesson(0).GetTopology().GetOutputCount() != network.GetPerceptronNetwork().GetTopology().GetFirstLayerTopology().GetNeuronCount().");
        }
      }
    }
  }
}