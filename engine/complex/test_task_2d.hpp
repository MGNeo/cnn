#pragma once

#include <stdexcept>

#include "i_test_task_2d.hpp"
#include "i_network_2d.hpp"
#include "i_lesson_2d.hpp"

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

        TestTask2D(const ILesson2D<T>& lesson,
                   INetwork2D<T>& network,
                   T& error);

        void Execute() override;

      private:

        const ILesson2D<T>& Lesson;
        INetwork2D<T>& Network;
        T& Error;

        void ProcessConvolutionNetwork();
        void ProcessPerceptronNetwork();
        T CalculateError();

      };

      template <typename T>
      TestTask2D<T>::TestTask2D(const ILesson2D<T>& lesson,
                                INetwork2D<T>& network,
                                T& error)
        :
        Lesson{ lesson },
        Network{ network },
        Error{ error }
      {
      }

      template <typename T>
      void TestTask2D<T>::Execute()
      {
        ProcessConvolutionNetwork();
        ProcessPerceptronNetwork();
        Error = CalculateError();
      }

      template <typename T>
      void TestTask2D<T>::ProcessConvolutionNetwork()
      {
        auto& convolutionNetwork = Network.GetConvolutionNetwork2D();
        auto& layer = convolutionNetwork.GetFirstLayer();
        for (size_t i = 0; i < Lesson.GetInputCount(); ++i)
        {
          const auto& lessonInput = Lesson.GetInput(i);
          auto& layerInput = layer.GetInput(i);
          for (size_t x = 0; x < Lesson.GetInputWidth(); ++x)
          {
            for (size_t y = 0; y < Lesson.GetInputHeight(); ++y)
            {
              layerInput.SetValue(x, y, lessonInput.GetValue(x, y));
            }
          }
        }
        convolutionNetwork.Process();
      }

      template <typename T>
      void TestTask2D<T>::ProcessPerceptronNetwork()
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
      T TestTask2D<T>::CalculateError()
      {
        T error{};
        const auto& perceptronNetwork = Network.GetPerceptronNetwork();
        const auto& perceptronOutput = perceptronNetwork.GetLastLayer().GetOutput();
        const auto& lessonOutput = Lesson.GetOutput();
        T maxError = 0;
        for (size_t o = 0; o < perceptronOutput.GetValueCount(); ++o)
        {
          error += std::abs(perceptronOutput.GetValue(o) - lessonOutput.GetValue(o));
        }
        return error;
      }
    }
  }
}