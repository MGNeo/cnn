#pragma once

#include "i_genetic_algorithm_2d.hpp"

#include <stdexcept>

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

        GeneticAlgorithm2D(const size_t populationSize,
                           const size_t iterationCount);

        size_t GetPopulationSize() const override;
        size_t GetIterationCount() const override;

        typename INetwork2D<T>::Uptr Run(const ILesson2DLibrary<T>& lessonLibrary,
                                         const INetwork2D<T>& network) const override;

      private:

        size_t PopulationSize;
        size_t IterationCount;

      };

      template <typename T>
      GeneticAlgorithm2D<T>::GeneticAlgorithm2D(const size_t populationSize,
                                                const size_t iterationCount)
        :
        PopulationSize{ populationSize },
        IterationCount{ iterationCount }
      {
        if (PopulationSize <= 4)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::GeneticAlgorithm2D(), PopulationSize <= 4.");
        }
        if (IterationCount == 0)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::GeneticAlgorithm2D(), IterationCount == 0.");
        }
      }

      template <typename T>
      size_t GeneticAlgorithm2D<T>::GetPopulationSize() const
      {
        return PopulationSize;
      }

      template <typename T>
      size_t GeneticAlgorithm2D<T>::GetIterationCount() const
      {
        return IterationCount;
      }

      template <typename T>
      typename INetwork2D<T>::Uptr GeneticAlgorithm2D<T>::Run(const ILesson2DLibrary<T>& lessonLibrary,
                                                              const INetwork2D<T>& network) const
      {
        // ...
      }
    }
  }
}