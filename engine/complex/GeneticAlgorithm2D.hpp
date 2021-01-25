#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#include "../complex/Network2D.hpp"
#include "../complex/Lesson2DLibrary.hpp"
#include "../common/ValueGenerator.hpp"
#include "../common/Mutagen.hpp"

#include "GeneticTest2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class GeneticAlgorithm2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        GeneticAlgorithm2D(const size_t threadCount = 0,
                           const size_t iterationCount = MIN_ITERATION_COUNT);

        GeneticAlgorithm2D(const GeneticAlgorithm2D& algorithm) = default;
        GeneticAlgorithm2D(GeneticAlgorithm2D&& algorithm) noexcept;

        GeneticAlgorithm2D& operator=(const GeneticAlgorithm2D& algorithm);
        GeneticAlgorithm2D& operator=(GeneticAlgorithm2D&& algorithm) noexcept;

        size_t GetThreadCount() const noexcept;
        void SetThreadCount(const size_t threadCount);

        size_t GetIterationCount() const noexcept;
        void SetIterationCount(const size_t iterationCount);

        const common::ValueGenerator<T>& GetValueGenerator() const noexcept;
        void SetValueGenerator(const common::ValueGenerator<T>& valueGenerator);

        const common::Mutagen<T>& GetMutagen() const noexcept;
        void SetMutagen(const common::Mutagen<T>& mutagen);

        void Clear() noexcept;

        Network2D<T> Run(const Lesson2DLibrary<T>& lessonLibrary, const Network2D<T>& sourceNetwork);

      private:

        constexpr static size_t MIN_ITERATION_COUNT = 10;

        size_t ThreadCount;
        size_t IterationCount;

        common::ValueGenerator<T> ValueGenerator;
        common::Mutagen<T> Mutagen;

        void CheckTopologies(const Lesson2DLibrary<T>& lessonLibrary, const Network2D<T>& sourceNetwork) const;

      };

      template <typename T>
      GeneticAlgorithm2D<T>::GeneticAlgorithm2D(const size_t threadCount,
                                                const size_t iterationCount)
        :
        ThreadCount{ threadCount },
        IterationCount{ iterationCount }
      {
      }

      template <typename T>
      GeneticAlgorithm2D<T>::GeneticAlgorithm2D(GeneticAlgorithm2D&& algorithm) noexcept
        :
        ThreadCount{ algorithm.ThreadCount },
        IterationCount{ algorithm.IterationCount },
        ValueGenerator{ std::move(algorithm.ValueGenerator) },
        Mutagen{ std::move(algorithm.Mutagen) }
      {
        algorithm.Reset();
      }
      
      template <typename T>
      GeneticAlgorithm2D<T>& GeneticAlgorithm2D<T>::operator=(const GeneticAlgorithm2D& algorithm)
      {
        if (this != &algorithm)
        {
          GeneticAlgorithm2D<T> tmpAlgorithm{ algorithm };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpAlgorithm);
        }
        return *this;
      }

      template <typename T>
      GeneticAlgorithm2D<T>& GeneticAlgorithm2D<T>::operator=(GeneticAlgorithm2D&& algorithm) noexcept
      {
        if (this != &algorithm)
        {
          ThreadCount = algorithm.ThreadCount;
          IterationCount = algorithm.IterationCount;
          ValueGenerator = std::move(algorithm.ValueGenerator);
          Mutagen = std::move(algorithm.Mutagen);

          algorithm.Reset();
        }
        return *this;
      }

      template <typename T>
      size_t GeneticAlgorithm2D<T>::GetThreadCount() const noexcept
      {
        return ThreadCount;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetThreadCount(const size_t threadCount) 
      {
        ThreadCount = threadCount;
      }

      template <typename T>
      size_t GeneticAlgorithm2D<T>::GetIterationCount() const noexcept
      {
        return IterationCount;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetIterationCount(const size_t iterationCount)
      {
        IterationCount = iterationCount;
      }

      template <typename T>
      const common::ValueGenerator<T>& GeneticAlgorithm2D<T>::GetValueGenerator() const noexcept
      {
        return ValueGenerator;
      }
      
      template <typename T>
      void GeneticAlgorithm2D<T>::SetValueGenerator(const common::ValueGenerator<T>& valueGenerator)
      {
        ValueGenerator = valueGenerator;
      }

      template <typename T>
      const common::Mutagen<T>& GeneticAlgorithm2D<T>::GetMutagen() const noexcept
      {
        return Mutagen;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetMutagen(const common::Mutagen<T>& mutagen)
      {
        Mutagen = mutagen;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::Clear() noexcept
      {
        ThreadCount = 0;
        IterationCount = MIN_ITERATION_COUNT;
        ValueGenerator.Clear();
        Mutagen.Clear();
      }

      template <typename T>
      Network2D<T> GeneticAlgorithm2D<T>::Run(const Lesson2DLibrary<T>& lessonLibrary,
                                              const Network2D<T>& sourceNetwork)
      {
        CheckTopologies(lessonLibrary, sourceNetwork);

        T bestError = std::numeric_limits<T>::max();
        Network2D<T> bestNetwork = sourceNetwork;

        for (size_t i = 0; i < IterationCount; ++i)
        {
          Network2D<T> newNetwork = bestNetwork;
          newNetwork.Mutate(Mutagen);

          GeneticTest2D<T> test(lessonLibrary,
                                newNetwork,
                                ThreadCount);

          if (test.GetTotalError() < bestError)
          {
            std::swap(bestNetwork, newNetwork);
            bestError = test.GetTotalError();
          }
        }

        return std::move(bestNetwork);
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::CheckTopologies(const Lesson2DLibrary<T>& lessonLibrary,
                                                  const Network2D<T>& sourceNetwork) const
      {
        if (lessonLibrary.GetLessonCount() == 0)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::CheckTopologies(), lessonLibrary.GetLessonCount() == 0.");
        }
        if (lessonLibrary.GetLesson(0).GetTopology().GetInputSize() != sourceNetwork.GetConvolutionNetwork().GetTopology().GetFirstLayerTopology().GetInputSize())
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::CheckTopologies(), lessonLibrary.GetLesson(0).GetTopology().GetInputSize() != sourceNetwork.GetConvolutionNetwork().GetTopology().GetFirstLayerTopology().GetInputSize().");
        }
        if (lessonLibrary.GetLesson(0).GetTopology().GetInputCount() != sourceNetwork.GetConvolutionNetwork().GetTopology().GetFirstLayerTopology().GetInputCount())
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::CheckTopologies(), lessonLibrary.GetLesson(0).GetTopology().GetInputCount() != sourceNetwork.GetConvolutionNetwork().GetTopology().GetFirstLayerTopology().GetInputCount().");
        }
        if (lessonLibrary.GetLesson(0).GetTopology().GetOutputCount() != sourceNetwork.GetPerceptronNetwork().GetTopology().GetFirstLayerTopology().GetNeuronCount())
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::CheckTopologies(), lessonLibrary.GetLesson(0).GetTopology().GetOutputCount() != sourceNetwork.GetPerceptronNetwork().GetTopology().GetFirstLayerTopology().GetNeuronCount().");
        }
      }
    }
  }
}