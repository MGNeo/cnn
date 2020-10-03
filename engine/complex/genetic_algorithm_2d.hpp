#pragma once

#include <stdexcept>
#include <vector>

#include "i_genetic_algorithm_2d.hpp"
#include "network_2d.hpp"
#include "../common/value_generator.hpp"
#include "../common/binary_random_generator.hpp"
#include "../common/mutagen.hpp"

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

        T GetMinWeightValue() const override;
        void SetMinWeightValue(const T minWeightValue) override;

        T GetMaxWeightValue() const override;
        void SetMaxWeightValue(const T maxWeightValue) override;

        size_t GetPopulationSize() const override;
        void SetPopulationSize(const size_t populationSize) override;

        size_t GetIterationCount() const override;
        void SetIterationCount(const size_t iterationCount) override;

        typename INetwork2D<T>::Uptr Run(const ILesson2DLibrary<T>& lessonLibrary,
                                         const INetwork2D<T>& network) const override;

      private:

        T MinWeightValue;
        T MaxWeightValue;
        size_t PopulationSize;
        size_t IterationCount;

        void Preparation(const complex::INetwork2D<T>& sourceNetwork,
                         std::vector<typename complex::INetwork2D<T>::Uptr>& sourcePopulation,
                         std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const;

        void Cross(std::vector<typename complex::INetwork2D<T>::Uptr>& sourcePopulation,
                   std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const;

        void Mutate(std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const;

      };

      template <typename T>
      GeneticAlgorithm2D<T>::GeneticAlgorithm2D()
        :
        MinWeightValue{ -1 },
        MaxWeightValue{ +1 },
        PopulationSize{ MIN_POPULATION_SIZE },
        IterationCount{ MIN_ITERATION_COUNT }
      {
      }

      template <typename T>
      T GeneticAlgorithm2D<T>::GetMinWeightValue() const
      {
        return MinWeightValue;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetMinWeightValue(const T minWeightValue)
      {
        if (MinWeightValue >= MaxWeightValue)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::SetMinWeight(), MinWeightValue >= MaxWeightValue.");
        }
        MinWeightValue = minWeightValue;
      }

      template <typename T>
      T GeneticAlgorithm2D<T>::GetMaxWeightValue() const
      {
        return MaxWeightValue;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetMaxWeightValue(const T maxWeightValue)
      {
        if (MaxWeightValue <= MinWeightValue)
        {
          throw std::invalid_argument("cnn::engine::complex::GeneticAlgorithm2D::SetMaxWeight(), MaxWeightValue <= MinWeightValue.");
        }
        MaxWeightValue = maxWeightValue;
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
                                                              const INetwork2D<T>& network) const
      {
        std::vector<typename complex::INetwork2D<T>::Uptr> sourcePopulation;
        std::vector<typename complex::INetwork2D<T>::Uptr> resultPopulation;

        Preparation(network, sourcePopulation, resultPopulation);

        for (size_t i = 0; i < GetIterationCount(); ++i)
        {
          Cross(sourcePopulation, resultPopulation);
          Mutate(resultPopulation);
        }

        return {};
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::Preparation(const complex::INetwork2D<T>& sourceNetwork,
                                              std::vector<typename complex::INetwork2D<T>::Uptr>& sourcePopulation,
                                              std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const
      {
        const size_t sourcePopulationSize = PopulationSize;
        const size_t m = sourcePopulationSize - 1;
        const size_t resultPopulationSize = sourcePopulationSize * m;

        if ((resultPopulationSize / sourcePopulationSize) != m)
        {
          throw std::overflow_error("cnn::engine::complex::GeneticAlgorithm2D::Preparation(), resultPopulationSize has been overflowed.");
        }

        sourcePopulation.resize(sourcePopulationSize);
        resultPopulation.resize(resultPopulationSize);

        auto valueGenerator = std::make_unique<common::ValueGenerator<T>>(GetMinWeightValue(), GetMaxWeightValue());

        sourcePopulation[0] = sourceNetwork.Clone(true);
        for (auto& network : sourcePopulation)
        {
          network = sourceNetwork.Clone(false);
          network->FillWeights(*valueGenerator);
        }
        for (auto& network : resultPopulation)
        {
          network = sourceNetwork.Clone(false);
        }
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::Cross(std::vector<typename complex::INetwork2D<T>::Uptr>& sourcePopulation,
                                        std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const
      {
        auto binaryRandomGenerator = std::make_unique<common::BinaryRandomGenerator>();

        size_t r{};
        for (const auto& sourceNetwork1 : sourcePopulation)
        {
          for (const auto& sourceNetwork2 : sourcePopulation)
          {
            if (sourceNetwork1 != sourceNetwork2)
            {
              resultPopulation[r++]->CrossFrom(*sourceNetwork1, *sourceNetwork2, *binaryRandomGenerator);
            }
          }
        }
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::Mutate(std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const
      {
        auto mutagen = std::make_unique<common::Mutagen<T>>();

        mutagen->SetMinResult(GetMinWeightValue());
        mutagen->SetMaxResult(GetMaxWeightValue());

        for (auto& network : resultPopulation)
        {
          // 0 TODO:
          // Write Mutate() method for all what have weights.
          //network->Mutation(mutagen);
        }
      }
    }
  }
}