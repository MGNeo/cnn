#pragma once

#include <cstddef>
#include <cstdint>

#include "../common/ValueGenerator.hpp"
#include "../common/Mutagen.hpp"

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

        GeneticAlgorithm2D();

        GeneticAlgorithm2D(const GeneticAlgorithm2D& algorithm) = default;
        GeneticAlgorithm2D(GeneticAlgorithm2D&& algorithm) noexcept;

        GeneticAlgorithm2D& operator=(const GeneticAlgorithm2D& algorithm) = default;
        GeneticAlgorithm2D& operator=(GeneticAlgorithm2D&& algorithm) noexcept;

        size_t GetThreadCount() const;
        void SetThreadCount(const size_t threadCount);

        size_t GetIterationCount() const;
        void SetIterationCount(const size_t iterationCount);

        const common::ValueGenerator<T>& GetValueGenerator() const;
        void SetValueGenerator(const common::ValueGenerator<T>& valueGenerator);

        const common::Mutagen<T>& GetMutagen() const;
        void SetMutagen(const common::Mutagen<T>& mutagen);

        //Network2D<T> Run(const Lesson2DLibrary<T>& lessonLibrary, const Network2D<T>& sourceNetwork);

      private:

        constexpr static size_t MIN_ITERATION_COUNT = 10;

        size_t ThreadCount;
        size_t IterationCount;

        common::ValueGenerator<T> ValueGenerator;
        common::Mutagen<T> Mutagen;

      };

      template <typename T>
      GeneticAlgorithm2D<T>::GeneticAlgorithm2D()
        :
        ThreadCount{ 0 },
        IterationCount{ MIN_ITERATION_COUNT }
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
        ThreadCount = 0;
        IterationCount = 0;
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

          algorithm.ThreadCount = 0;
          algorithm.IterationCount = 0;
        }
        return *this;
      }

      template <typename T>
      size_t GeneticAlgorithm2D<T>::GetThreadCount() const
      {
        return ThreadCount;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetThreadCount(const size_t threadCount)
      {
        ThreadCount = threadCount;
      }

      template <typename T>
      size_t GeneticAlgorithm2D<T>::GetIterationCount() const
      {
        return IterationCount;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetIterationCount(const size_t iterationCount)
      {
        IterationCount = iterationCount;
      }

      template <typename T>
      const common::ValueGenerator<T>& GeneticAlgorithm2D<T>::GetValueGenerator() const
      {
        return ValueGenerator;
      }
      
      template <typename T>
      void GeneticAlgorithm2D<T>::SetValueGenerator(const common::ValueGenerator<T>& valueGenerator)
      {
        ValueGenerator = valueGenerator;
      }

      template <typename T>
      const common::Mutagen<T>& GeneticAlgorithm2D<T>::GetMutagen() const
      {
        return Mutagen;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetMutagen(const common::Mutagen<T>& mutagen)
      {
        Mutagen = mutagen;
      }
    }
  }
}