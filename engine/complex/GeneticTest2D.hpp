#pragma once

#include "Lesson2DLibrary.hpp"
#include "Network2D.hpp"

#include <thread>

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class GeneticTest2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        GeneticTest2D(const Lesson2DLibrary<T>& lessonLibrary,
                      const Network2D<T>& network,
                      const size_t threadCount = 0);

        T GetTotalError() const noexcept;

      private:
        
        const T TotalError;

      };

      template <typename T>
      GeneticTest2D<T>::GeneticTest2D(const Lesson2DLibrary<T>& lessonLibrary,
        const Network2D<T>& network,
        const size_t threadCount)
      {
        const size_t count = threadCount ? threadCount : std::thread::hardware_concurrency();

        // Copy network.
        std::vector<Network2D<T>> networks(count, networks);

        // Run threads.

        // Wait the threads.
      }

      template <typename T>
      T GeneticTest2D<T>::GetTotalError() const noexcept
      {
        return TotalError;
      }

    }
  }
}