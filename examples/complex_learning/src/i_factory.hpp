#pragma once

#include <memory>
#include <cstdint>
#include <cstddef>
#include <string>

#include "engine/complex/i_lesson_2d_library.hpp"
#include "engine/complex/i_network_2d.hpp"
#include "engine/complex/i_genetic_algorithm_2d.hpp"

namespace cnn
{
  namespace examples
  {
    namespace complex_learning
    {
      template <typename T>
      class IFactory
      {

        static_assert(std::is_floating_point<T>::value);
        
      public:

        using Uptr = std::unique_ptr<IFactory<T>>;

        // The result must not be nullptr.
        virtual typename engine::complex::ILesson2DLibrary<T>::Uptr Library() const = 0;
        // The result must not be nullptr.
        virtual typename engine::complex::INetwork2D<T>::Uptr Network() const = 0;
        // The result must not be nullptr.
        virtual typename engine::complex::IGeneticAlgorithm2D<T>::Uptr Algorithm() const = 0;

        virtual ~IFactory() = default;

      };
    }
  }
}