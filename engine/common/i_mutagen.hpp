#pragma once

#include <memory>
#include <cstddef>
#include <cstdint>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class IMutagen
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<IMutagen<T>>;

        virtual T Mutate(const T value) = 0;

        virtual typename IMutagen<T>::Uptr Clone() const = 0;

        virtual ~IMutagen() = default;

      };
    }
  }
}