#pragma once

#include "Layer2D.hpp"
#include "ProxyLayer2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Network2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

      private:

        //Network2DTopology Topology;
        //std::unique_ptr<Layer2D<T>[]> Layers;

      };
    }
  }
}