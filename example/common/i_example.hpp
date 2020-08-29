#pragma once

#include <type_traits>
#include <memory>

namespace cnn
{
  namespace example
  {
    namespace common
    {
      template <typename T>
      class IExample
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<IExample<T>>;

        virtual void Execute() const = 0;

        virtual ~IExample() {};

      };
    }
  }
}
