#pragma once

#include <type_traits>
#include <memory>

#include "../common/i_map.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      template <typename T>
      class ILayer
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<ILayer<T>>;
        
        virtual size_t GetInputSize() const = 0;

        virtual const common::IMap<T>& GetInput() const = 0;
        virtual common::IMap<T>& GetInput() = 0;

        virtual size_t GetOutputSize() const = 0;

        virtual const common::IMap<T>& GetOutput() const = 0;
        virtual common::IMap<T>& GetOutput() = 0;

        virtual void Process() = 0;

        virtual ~ILayer() {}

      };
    }
  }
}