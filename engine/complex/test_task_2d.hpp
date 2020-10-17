#pragma once

#include <stdexcept>

#include "i_test_task_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class TestTask2D : public ITestTask2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        TestTask2D(const ILesson2DLibrary<T>& library,
                   INetwork2D<T>& network,
                   T& error);

        void Execute() override;

      private:

        INetwork2D<T>& Network;
        const ILesson2DLibrary<T>& Library;
        T& Error;

        void ProcessConvolutionNetwork(const size_t lessonNumber);
        void ProcessPerceptronNetwork(const size_t lessonNumber);
        T CalculateError(const size_t lessonNumber);

      };

      template <typename T>
      TestTask2D<T>::TestTask2D(const ILesson2DLibrary<T>& library,
                                INetwork2D<T>& network,
                                T& error)
        :
        Library{ library },
        Network{ network },
        Error{ error }
      {
      }

      template <typename T>
      void TestTask2D<T>::Execute()
      {
        T totalError{};
        for (size_t l = 0; l < Library.GetLessonCount(); ++l)
        {
          ProcessConvolutionNetwork(l);
          ProcessPerceptronNetwork(l);
          totalError += CalculateError(l);
        }
        Error = totalError;
      }

      template <typename T>
      void TestTask2D<T>::ProcessConvolutionNetwork(const size_t lessonNumber)
      {
        auto& convolutionNetwork = Network.GetConvolutionNetwork2D();
        auto& layer = convolutionNetwork.GetFirstLayer();
        const auto& lesson = Library.GetLesson(lessonNumber);
        for (size_t i = 0; i < Library.GetLessonInputCount(); ++i)
        {
          const auto& lessonInput = lesson.GetInput(i);
          auto& layerInput = layer.GetInput(i);
          for (size_t x = 0; x < Library.GetLessonInputWidth(); ++x)
          {
            for (size_t y = 0; y < Library.GetLessonInputHeight(); ++y)
            {
              layerInput.SetValue(x, y, lessonInput.GetValue(x, y));
            }
          }
        }
        convolutionNetwork.Process();
      }

      template <typename T>
      void TestTask2D<T>::ProcessPerceptronNetwork(const size_t lessonNumber)
      {
        const auto& convolutionNetwork = Network.GetConvolutionNetwork2D();
        auto& perceptronNetwork = Network.GetPerceptronNetwork();
        auto& firstLayer = perceptronNetwork.GetFirstLayer();
        const auto& lastLayer = convolutionNetwork.GetLastLayer();
        size_t i{};
        for (size_t o = 0; o < lastLayer.GetOutputCount(); ++o)
        {
          auto& input = firstLayer.GetInput();
          const auto& output = lastLayer.GetOutput(o);
          for (size_t x = 0; x < lastLayer.GetOutputWidth(); ++x)
          {
            for (size_t y = 0; y < lastLayer.GetOutputHeight(); ++y)
            {
              input.SetValue(i++, output.GetValue(x, y));
            }
          }
        }
        perceptronNetwork.Process();
      }

      template <typename T>
      T TestTask2D<T>::CalculateError(const size_t lessonNumber)
      {
        T error{};
        const auto& perceptronNetwork = Network.GetPerceptronNetwork();
        const auto& perceptronOutput = perceptronNetwork.GetLastLayer().GetOutput();
        const auto& lessonOutput = Library.GetLesson(lessonNumber).GetOutput();
        for (size_t o = 0; o < perceptronOutput.GetValueCount(); ++o)
        {
          error += std::abs(perceptronOutput.GetValue(o) - lessonOutput.GetValue(o));
        }
        return error;
      }
    }
  }
}