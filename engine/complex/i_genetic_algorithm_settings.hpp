#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>

#include "i_network_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class IGeneticAlgorithmSettings
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        virtual const INetwork2D<T>& GetNetwork() const = 0;

        virtual ~IGeneticAlgorithmSettings() = default;

      }
    }
  }
}