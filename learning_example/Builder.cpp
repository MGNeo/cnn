#include "Builder.hpp"

namespace cnn
{
  namespace learning_example
  {
    template <typename T>
    typename engine::complex::Lesson2DLibrary<T>::Uptr Builder<T>::GetLessonLibrary()
    {
      // ...
      return {};
    }

    template <typename T>
    typename engine::complex::Network2D<T>::Uptr Builder<T>::GetNetwork()
    {
      // ...
      return {};
    }

    template <typename T>
    typename engine::complex::GeneticAlgorithm2D<T>::Uptr Builder<T>::GetGeneticAlgorithm()
    {
      // ...
      return {};
    }
  }
}