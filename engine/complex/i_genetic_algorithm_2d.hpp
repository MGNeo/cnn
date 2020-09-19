#pragma once

#include <type_traits>
#include <memory>

#include "i_network_2d.hpp"
#include "i_lesson_2d_library.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class IGeneticAlgorithm2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<IGeneticAlgorithm2D<T>>;

        virtual T GetMinWeightValue() const = 0;
        virtual void SetMinWeightValue(const T minWeightValue) = 0;

        virtual T GetMaxWeightValue() const = 0;
        virtual void SetMaxWeightValue(const T maxWeightValue) = 0;

        virtual size_t GetPopulationSize() const = 0;
        virtual void SetPopulationSize(const size_t populationSize) = 0;

        virtual size_t GetIterationCount() const = 0;
        virtual void SetIterationCount(const size_t iterationCount) = 0;

        virtual typename INetwork2D<T>::Uptr Run(const ILesson2DLibrary<T>& lessonLibrary,
                                                 const INetwork2D<T>& network) const = 0;

        virtual ~IGeneticAlgorithm2D() = default;

      };
    }
  }
}