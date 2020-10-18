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

        virtual T GetMinResult() const = 0;
        virtual void SetMinResult(const T minResult) = 0;

        virtual T GetMaxResult() const = 0;
        virtual void SetMaxResult(const T maxResult) = 0;

        virtual T GetVariabilityForce() const = 0;
        virtual void SetVariabilityForce(const T variabilityForce) = 0;

        virtual T Mutate(const T value) = 0;

        virtual typename IMutagen<T>::Uptr Clone() const = 0;

        virtual ~IMutagen() = default;

      };
    }
  }
}