#pragma once

#include <stdexcept>
#include <vector>

#include "i_genetic_algorithm_2d.hpp"
#include "network_2d.hpp"
#include "../common/value_generator.hpp"

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

        static constexpr size_t MIN_POPULATION_SIZE = 10;
        static constexpr size_t MIN_ITERATION_COUNT = 10;

        GeneticAlgorithm2D();

        T GetMinWeight() const override;
        void SetMinWeight(const T minWeight) override;

        T GetMaxWeight() const override;
        void SetMaxWeight(const T maxWeight) override;

        size_t GetPopulationSize() const override;
        void SetPopulationSize(const size_t populationSize) override;

        size_t GetIterationCount() const override;
        void SetIterationCount(const size_t iterationCount) override;

        typename INetwork2D<T>::Uptr Run(const ILesson2DLibrary<T>& lessonLibrary,
                                         const INetwork2D<T>& network) override;

      private:

        T MinWeight;
        T MaxWeight;
        size_t PopulationSize;
        size_t IterationCount;

      };

      template <typename T>
      GeneticAlgorithm2D<T>::GeneticAlgorithm2D()
        :
        MinWeight{ -1 },
        MaxWeight{ +1 },
        PopulationSize{ MIN_POPULATION_SIZE },
        IterationCount{ MIN_ITERATION_COUNT }
      {
      }

      template <typename T>
      T GeneticAlgorithm2D<T>::GetMinWeight() const
      {
        return MinWeight;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetMinWeight(const T minWeight)
      {
        if (MinWeight >= MaxWeight)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::SetMinWeight(), MinWeight >= MaxWeight.");
        }
        MinWeight = minWeight;
      }

      template <typename T>
      T GeneticAlgorithm2D<T>::GetMaxWeight() const
      {
        return MaxWeight;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetMaxWeight(const T maxWeight)
      {
        if (MaxWeight <= MinWeight)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::SetMaxWeight(), MaxWeight <= MinWeight.");
        }
        MaxWeight = maxWeight;
      }

      template <typename T>
      size_t GeneticAlgorithm2D<T>::GetPopulationSize() const
      {
        return PopulationSize;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetPopulationSize(const size_t populationSize)
      {
        if (populationSize < MIN_POPULATION_SIZE)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::SetPopulationSize(), populationSize < MIN_POPULATION_SIZE.");
        }
        PopulationSize = populationSize;
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
      typename INetwork2D<T>::Uptr GeneticAlgorithm2D<T>::Run(const ILesson2DLibrary<T>& lessonLibrary,
                                                              const INetwork2D<T>& network)
      {
        const size_t sourcePopulationSize = PopulationSize;
        const size_t resultPopulationSize = sourcePopulationSize * sourcePopulationSize;

        if ((resultPopulationSize / sourcePopulationSize) != sourcePopulationSize)
        {
          throw std::overflow_error("cnn::engine::complex::GeneticAlgorithm2D::Run(), resultPopulationSize has been overflowed.");
        }

        auto sourcePopulation = std::make_unique<typename INetwork2D<T>::Uptr[]>(sourcePopulationSize);
        auto resultPopulation = std::make_unique<typename INetwork2D<T>::Uptr[]>(resultPopulationSize);

        // TODO: ...

        return {};
      }

    }
  }
}