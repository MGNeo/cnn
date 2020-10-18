#pragma once

#include <type_traits>
#include <memory>

#include "i_network_2d.hpp"
#include "i_lesson_2d_library.hpp"
#include "../common/i_value_generator.hpp"
#include "../common/i_mutagen.hpp"

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

        // The mutagen is used for mutating the weights.
        virtual const typename common::IMutagen<T>& GetMutagen() const = 0;
        virtual void SetMutagen(const common::IMutagen<T>& mutagen) = 0;

        virtual size_t GetIterationCount() const = 0;
        virtual void SetIterationCount(const size_t iterationCount) = 0;

        virtual size_t GenThreadCount() const = 0;
        virtual void SetThreadCount(const size_t threadCount) = 0;

        virtual typename INetwork2D<T>::Uptr Run(const ILesson2DLibrary<T>& lessonLibrary,
                                                 const INetwork2D<T>& network) const = 0;

        virtual ~IGeneticAlgorithm2D() = default;

      };
    }
  }
}