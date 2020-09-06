#pragma once

#include "i_genetic_algorithm_2d.hpp"

#include "network_2d.hpp"

#include <stdexcept>
#include <vector>

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

        GeneticAlgorithm2D(const size_t sourcePopulationSize,
                           const size_t iterationCount);

        size_t GetSourcePopulationSize() const override;
        size_t GetIterationCount() const override;

        typename INetwork2D<T>::Uptr Run(const ILesson2DLibrary<T>& lessonLibrary,
                                         const INetwork2D<T>& network) override;

      private:

        size_t SourcePopulationSize;
        std::unique_ptr<typename INetwork2D<T>::Uptr[]> SourcePopulation;

        size_t ResultPopulationSize;
        std::unique_ptr<typename INetwork2D<T>::Uptr[]> ResultPopulation;

        size_t IterationCount;

        void ClearPopulations();
        void NoisePopulations();

      };

      template <typename T>
      GeneticAlgorithm2D<T>::GeneticAlgorithm2D(const size_t sourcePopulationSize,
                                                const size_t iterationCount)
        :
        SourcePopulationSize{ sourcePopulationSize },
        IterationCount{ iterationCount }
      {
        if (SourcePopulationSize <= 4)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::GeneticAlgorithm2D(), SourcePopulationSize <= 4.");
        }

        {
          const size_t s = SourcePopulationSize - 1;
          ResultPopulationSize = SourcePopulationSize * s;
          if ((ResultPopulationSize / SourcePopulationSize) != s)
          {
            throw std::overflow_error("cnn::engine::complex::GeneticAlgorithm2D::GeneticAlgorithm2D(), ResultPopulationSize was overflowed.");
          }
        }

        if (IterationCount == 0)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::GeneticAlgorithm2D(), IterationCount == 0.");
        }

        SourcePopulation = std::make_unique<typename INetwork2D<T>::Uptr[]>(SourcePopulationSize);
        ResultPopulation = std::make_unique<typename INetwork2D<T>::Uptr[]>(ResultPopulationSize);
      }

      template <typename T>
      size_t GeneticAlgorithm2D<T>::GetSourcePopulationSize() const
      {
        return SourcePopulationSize;
      }

      template <typename T>
      size_t GeneticAlgorithm2D<T>::GetIterationCount() const
      {
        return IterationCount;
      }

      template <typename T>
      typename INetwork2D<T>::Uptr GeneticAlgorithm2D<T>::Run(const ILesson2DLibrary<T>& lessonLibrary,
                                                              const INetwork2D<T>& network)
      {
        ClearPopulations();
        NoisePopulations();
        return{};
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::ClearPopulations()
      {
        for (size_t s = 0; s < SourcePopulationSize; ++s)
        {
          SourcePopulation[s] = nullptr;
        }
        for (size_t r = 0; r < ResultPopulationSize; ++r)
        {
          ResultPopulationSize[r] = nullptr;
        }
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::NoisePopulations()
      {
        // ...
      }
    }
  }
}