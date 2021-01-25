#pragma once

#include <type_traits>

#include "../engine/complex/Lesson2DLibrary.hpp"
#include "../engine/complex/Network2D.hpp"
#include "../engine/complex/GeneticAlgorithm2D.hpp"

namespace cnn
{
  namespace learning_example
  {
    template <typename T>
    class Builder
    {

      static_assert(std::is_floating_point<T>::value);

    public:

      static typename engine::complex::Lesson2DLibrary<T>::Uptr GetLessonLibrary();
      static typename engine::complex::Network2D<T>::Uptr GetNetwork();
      static typename engine::complex::GeneticAlgorithm2D<T>::Uptr GetGeneticAlgorithm();

    private:

      ~Builder() = delete;

    };
  }
}
