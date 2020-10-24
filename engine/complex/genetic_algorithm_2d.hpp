#pragma once

#include <stdexcept>
#include <vector>
#include <map>

#include "i_genetic_algorithm_2d.hpp"
#include "test_task_2d.hpp"
#include "network_2d.hpp"
#include "../common/value_generator.hpp"
#include "../common/binary_random_generator.hpp"
#include "../common/mutagen.hpp"
#include "test_task_2d.hpp"
#include "test_task_2d_pool.hpp"
#include "test_task_2d_thread_pool.hpp"
#include "../common/value_generator.hpp"
#include "../common/mutagen.hpp"

#include <iostream>// DEBUG
#include <list>

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class GeneticAlgorithm2D : public IGeneticAlgorithm2D<T>
      {

        static_assert(std::is_floating_point<T>::value);
          
      public:
        
        using Uptr = std::unique_ptr<GeneticAlgorithm2D<T>>;

        GeneticAlgorithm2D();

        const typename common::IMutagen<T>& GetMutagen() const override;
        void SetMutagen(const common::IMutagen<T>& mutagen) override;

        size_t GetIterationCount() const override;
        void SetIterationCount(const size_t iterationCount) override;

        size_t GenThreadCount() const override;
        void SetThreadCount(const size_t threadCount) override;

        typename INetwork2D<T>::Uptr Run(const ILesson2DLibrary<T>& lessonLibrary,
                                         const INetwork2D<T>& network) const override;

      private:

        static constexpr size_t MIN_ITERATION_COUNT = 10;

        typename common::IValueGenerator<T>::Uptr ValueGenerator;
        typename common::IMutagen<T>::Uptr Mutagen;

        size_t IterationCount;
        size_t ThreadCount;

        void Check(const ILesson2DLibrary<T>& lessonLibrary,
                   const INetwork2D<T>& network) const;

      };

      template <typename T>
      GeneticAlgorithm2D<T>::GeneticAlgorithm2D()
        :
        Mutagen{ std::make_unique<common::Mutagen<T>>() },

        IterationCount{ MIN_ITERATION_COUNT },
        ThreadCount{ 1 }
      {
      }

      template <typename T>
      const typename common::IMutagen<T>& GeneticAlgorithm2D<T>::GetMutagen() const
      {
        return *Mutagen;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetMutagen(const common::IMutagen<T>& mutagen)
      {
        Mutagen = mutagen.Clone();
      }

      template <typename T>
      size_t GeneticAlgorithm2D<T>::GetIterationCount() const
      {
        return IterationCount;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetIterationCount(const size_t iterationCount)
      {
        if (iterationCount < MIN_ITERATION_COUNT)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::SetIterationCount(), iterationCount < MIN_ITERATION_COUNT.");
        }
        IterationCount = iterationCount;
      }

      template <typename T>
      size_t GeneticAlgorithm2D<T>::GenThreadCount() const
      {
        return ThreadCount;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetThreadCount(const size_t threadCount)
      {
        if (threadCount == 0)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::SetThreadCount(), threadCOunt == 0.");
        }
        ThreadCount = threadCount;
      }

      template <typename T>
      typename INetwork2D<T>::Uptr GeneticAlgorithm2D<T>::Run(const ILesson2DLibrary<T>& lessonLibrary,
                                                              const INetwork2D<T>& network) const
      {
        Check(lessonLibrary, network);

        T bestError = std::numeric_limits<T>::max();

        auto bestNetwork = network.Clone(true);

        for (size_t i = 0; i < GetIterationCount(); ++i)
        {
          clock_t t1 = clock();// DEBUG

          // Clone the best network.
          auto newNetwork = bestNetwork->Clone(true);
          // Mutate the new network.
          newNetwork->Mutate(*Mutagen);
          // Test new network.
          {
            std::vector<T> errors(lessonLibrary.GetLessonCount());
            // Prepare the task pool.
            auto taskPool = std::make_unique<TestTask2DPool<T>>();
            for (size_t l = 0; l < lessonLibrary.GetLessonCount(); ++l)
            {
              const auto& lesson = lessonLibrary.GetLesson(l);
              auto task = std::make_unique<TestTask2D<T>>(lesson, *newNetwork, errors[l]);
              taskPool->Push(std::move(task));
            }
            // Prepare the thread pool.
            auto threadPool = std::make_unique<TestTask2DThreadPool<T>>(*taskPool, ThreadCount);

            // Wait the thread pool.
            threadPool->Wait();

            // Sum the errors.
            T newError{};
            for (const auto& error : errors)
            {
              newError += error;
            }
            
            // Compare.
            if (newError < bestError)
            {
              bestError = newError;
              bestNetwork.swap(newNetwork);
            }
          }
          
          clock_t t2 = clock();// DEBUG
          std::cout << "Time elapsed: " << (t2 - t1) / static_cast<float>(CLOCKS_PER_SEC)  << " Best error: " << bestError << std::endl;// DEBUG
        }

        return std::move(bestNetwork);
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::Check(const ILesson2DLibrary<T>& lessonLibrary,
                                        const INetwork2D<T>& network) const
      {
        if (lessonLibrary.GetLessonInputCount() != network.GetConvolutionNetwork2D().GetFirstLayer().GetInputCount())
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::Check(), lessonLibrary.GetLessonInputCount() != network.GetConvolutionNetwork2D().GetFirstLayer().GetInputCount().");
        }
        if (lessonLibrary.GetLessonInputWidth() != network.GetConvolutionNetwork2D().GetFirstLayer().GetInputWidth())
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::Check(), lessonLibrary.GetLessonInputWidth() != network.GetConvolutionNetwork2D().GetFirstLayer().GetInputWidth().");
        }
        if (lessonLibrary.GetLessonInputHeight() != network.GetConvolutionNetwork2D().GetFirstLayer().GetInputHeight())
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::Check(), lessonLibrary.GetLessonInputHeight() != network.GetConvolutionNetwork2D().GetFirstLayer().GetInputHeight().");
        }
        if (lessonLibrary.GetLessonOutputSize() != network.GetPerceptronNetwork().GetLastLayer().GetOutputSize())
        {
          throw std::invalid_argument("cnn::engine::Complex::GeneticAlgorithm2D::Check(), lessonLibrary.GetLessonOutputSize() != network.GetPerceptronNetwork().GetLastLayer().GetOutputSize().");
        }
      }
    }
  }
}