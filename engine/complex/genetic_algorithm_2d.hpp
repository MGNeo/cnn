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
        Check(lessonLibrary, network);

        std::vector<typename complex::INetwork2D<T>::Uptr> sourcePopulation;
        std::vector<typename complex::INetwork2D<T>::Uptr> resultPopulation;

        std::cout << "Prepare..." << std::endl;// DEBUG
        Prepare(network, sourcePopulation, resultPopulation);

        for (size_t i = 0; i < GetIterationCount(); ++i)
        {
          std::cout << "Cross..." << std::endl;// DEBUG
          Cross(sourcePopulation, resultPopulation);
          std::cout << "Mutate..." << std::endl;// DEBUG
          Mutate(resultPopulation);
          std::cout << "Test..." << std::endl;// DEBUG
          Test(lessonLibrary, resultPopulation);// This operation is the heaviest, it must be multithreaded in the first place (and without any synchronisations).
          std::cout << "Select..." << std::endl;// DEBUG
          Select(sourcePopulation, resultPopulation);
        }

        return {};
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
          network->Mutate(*mutagen);
        }
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::Test(const ILesson2DLibrary<T>& lessonLibrary,
                                       std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const
      {
        auto taskPool = std::make_unique<TestTask2DPool<T>>();
        for (const auto& network : resultPopulation)
        {
          auto task = std::make_unique<TestTask2D<T>>(lessonLibrary, *network);
          taskPool->Push(std::move(task));
        }

        auto threadPool = std::make_unique<TestTask2DThreadPool<T>>(*taskPool);

        while (true)
        {
          if (threadPool->IsWrong() == true)
          {
            throw std::runtime_error("cnn::engine::complex::GeneticAlgorithm2D::Test(), threadPool->IsWrong() == true.");
          }
          if (taskPool->IsEmpty() == true)
          {
            break;
          }
          std::this_thread::yield();
        }

        // TODO: Return test results from test-thread to this thread.
        // ...
      }

      template <typename T>
      void GeneticAlgorithm2D<T>::Select(std::vector<typename complex::INetwork2D<T>::Uptr>& sourcePopulation,
                                         std::vector<typename complex::INetwork2D<T>::Uptr>& resultPopulation) const
      {
        for (size_t s = 0; s < sourcePopulation.size(); ++s)
        {
          sourcePopulation[s].swap(resultPopulation[s]);
        }
      }

    }
  }
}