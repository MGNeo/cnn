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

        const typename common::IValueGenerator<T>& GetValueGenerator() const override;
        void SetValueGenerator(const common::IValueGenerator<T>& valueGenerator) override;

        const typename common::IMutagen<T>& GetMutagen() const override;
        void SetMutagen(const common::IMutagen<T>& mutagen) override;

        size_t GetPopulationSize() const override;
        void SetPopulationSize(const size_t populationSize) override;

        size_t GetIterationCount() const override;
        void SetIterationCount(const size_t iterationCount) override;

        typename INetwork2D<T>::Uptr Run(const ILesson2DLibrary<T>& lessonLibrary,
                                         const INetwork2D<T>& network) const override;

      private:

        static constexpr size_t MIN_POPULATION_SIZE = 3;
        static constexpr size_t MIN_ITERATION_COUNT = 1;

        typename common::IValueGenerator<T>::Uptr ValueGenerator;
        typename common::IMutagen<T>::Uptr Mutagen;

        size_t PopulationSize;
        size_t IterationCount;

        void Check(const ILesson2DLibrary<T>& lessonLibrary,
                   const INetwork2D<T>& network) const;

        void Prepare(const complex::INetwork2D<T>& sourceNetwork,
                     std::vector<typename complex::INetwork2D<T>::Uptr>& sourcePopulation,
                     std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const;

        void Cross(std::vector<typename complex::INetwork2D<T>::Uptr>& sourcePopulation,
                   std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const;

        void Mutate(std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const;

        void Test(const ILesson2DLibrary<T>& lessonLibrary,
                  std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const;

        void Select(std::vector<typename complex::INetwork2D<T>::Uptr>& sourcePopulation,
                    std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const;

      };

      template <typename T>
      GeneticAlgorithm2D<T>::GeneticAlgorithm2D()
        :
        ValueGenerator{ std::make_unique<common::ValueGenerator<T>>() },
        Mutagen{ std::make_unique<common::Mutagen<T>>() },

        PopulationSize{ MIN_POPULATION_SIZE },
        IterationCount{ MIN_ITERATION_COUNT }
      {
      }

      template <typename T>
      const typename common::IValueGenerator<T>& GeneticAlgorithm2D<T>::GetValueGenerator() const
      {
        return *ValueGenerator;
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::SetValueGenerator(const common::IValueGenerator<T>& valueGenerator)
      {
        ValueGenerator = valueGenerator.Clone();
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
        Check(lessonLibrary, network);

        std::vector<typename complex::INetwork2D<T>::Uptr> sourcePopulation;
        std::vector<typename complex::INetwork2D<T>::Uptr> resultPopulation;

        Prepare(network, sourcePopulation, resultPopulation);

        for (size_t i = 0; i < GetIterationCount(); ++i)
        {
          Cross(sourcePopulation, resultPopulation);
          Mutate(resultPopulation);
          Test(lessonLibrary, resultPopulation);
          Select(sourcePopulation, resultPopulation);
          std::cout << std::endl;// DEBUG
        }

        return std::move(sourcePopulation[0]);
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

      template <typename T>
      void GeneticAlgorithm2D<T>::Prepare(const complex::INetwork2D<T>& sourceNetwork,
                                          std::vector<typename complex::INetwork2D<T>::Uptr>& sourcePopulation,
                                          std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const
      {
        const size_t sourcePopulationSize = PopulationSize;
        const size_t m = sourcePopulationSize - 1;
        const size_t resultPopulationSize = sourcePopulationSize * m;

        if ((resultPopulationSize / sourcePopulationSize) != m)
        {
          throw std::overflow_error("cnn::engine::complex::GeneticAlgorithm2D::Prepare(), resultPopulationSize has been overflowed.");
        }

        sourcePopulation.resize(sourcePopulationSize);
        resultPopulation.resize(resultPopulationSize);

        sourcePopulation[0] = sourceNetwork.Clone(true);
        for (size_t n = 1; n < sourcePopulation.size(); ++n)
        {
          sourcePopulation[n] = sourceNetwork.Clone(false);
          sourcePopulation[n]->FillWeights(*ValueGenerator);
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
        for (auto& network : resultPopulation)
        {
          network->Mutate(*Mutagen);
        }
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::Test(const ILesson2DLibrary<T>& lessonLibrary,
                                       std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const
      {
        // Prepare the storage for result of the testing.
        std::vector<T> errors(resultPopulation.size());

        // Prepare the task pool.
        auto taskPool = std::make_unique<TestTask2DPool<T>>();
        for (size_t i = 0; i < resultPopulation.size(); ++i)
        {
          auto task = std::make_unique<TestTask2D<T>>(lessonLibrary, *(resultPopulation[i]), errors[i]);
          taskPool->Push(std::move(task));
        }

        // Prepare the thread pool.
        auto threadPool = std::make_unique<TestTask2DThreadPool<T>>(*taskPool);

        // Wait the thread pool.
        threadPool->Wait();

        // Sort the networks using sorting tree.
        std::multimap<T, size_t> sortingTree;
        for (size_t i = 0; i < resultPopulation.size(); ++i)
        {
          sortingTree.insert({ errors[i], i });
        }
        size_t i{};
        for (auto& sortedNode : sortingTree)
        {
          std::cout << sortedNode.first << " ";// DEBUG
          resultPopulation[i++].swap(resultPopulation[sortedNode.second]);
        }
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::Select(std::vector<typename complex::INetwork2D<T>::Uptr>& sourcePopulation,
                                         std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const
      {
        // TODO: Improve this algorithm.
        for (size_t s = 0; s < sourcePopulation.size(); ++s)
        {
          sourcePopulation[s].swap(resultPopulation[s]);
        }
      }

    }
  }
}