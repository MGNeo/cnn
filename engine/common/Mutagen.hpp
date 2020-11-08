#pragma once

#include <random>
#include <stdexcept>
#include <cstddef>
#include <cstdint>
#include <chrono>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class Mutagen
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Mutagen();

        Mutagen(const Mutagen& mutagen) = default;

        Mutagen(Mutagen&& mutagen) noexcept;

        Mutagen& operator=(const Mutagen& mutagen) = default;

        Mutagen& operator=(Mutagen&& mutagen) noexcept;

        T GetMinResult() const noexcept;

        // Exception guarantee: strong for this.
        void SetMinResult(const T minResult);

        T GetMaxResult() const noexcept;

        // Exception guarantee: strong for this.
        void SetMaxResult(const T maxResult);

        T GetVariabilityForce() const noexcept;

        // Exception guarantee: strong for this.
        void SetVariabilityForce(const T variabilityForce);

        // We expect that the method throws an exception never.
        void SetSeed(const unsigned int seed) noexcept;

        // We expect that the method throws an exception never.
        void Clear() noexcept;

        // We expect that the method throws an exception never.
        T Mutate(const T value) noexcept;

      private:

        T MinResult;

        T MaxResult;

        T VariabilityForce;

        std::default_random_engine DRE;

      };

      template <typename T>
      Mutagen<T>::Mutagen()
      {
        Clear();
      }

      template <typename T>
      Mutagen<T>::Mutagen(Mutagen&& mutagen) noexcept
        :
        MinResult{ mutagen.MinResult },
        MaxResult{ mutagen.MaxResult },
        VariabilityForce{ mutagen.VariabilityForce },
        DRE{ std::move(mutagen.DRE) }
      {
        mutagen.Clear();
      }

      template <typename T>
      Mutagen<T>& Mutagen<T>::operator=(Mutagen&& mutagen) noexcept
      {
        if (this != &mutagen)
        {
          MinResult = mutagen.MinResult;
          MaxResult = mutagen.MaxResult;
          VariabilityForce = mutagen.VariabilityForce;
          DRE = std::move(mutagen.DRE);

          mutagen.Clear();
        }
        return *this;
      }

      template <typename T>
      T Mutagen<T>::GetMinResult() const noexcept
      {
        return MinResult;
      }

      template <typename T>
      void Mutagen<T>::SetMinResult(const T minResult)
      {
        if (minResult > MaxResult)
        {
          throw std::invalid_argument("cnn::engine::common::Mutagen::SetMinResult(), minResult > MaxResult.");
        }
        MinResult = minResult;
      }

      template <typename T>
      T Mutagen<T>::GetMaxResult() const noexcept
      {
        return MaxResult;
      }

      template <typename T>
      void Mutagen<T>::SetMaxResult(const T maxResult)
      {
        if (maxResult < MinResult)
        {
          throw std::invalid_argument("cnn::engine::common::Mutagen::SetMaxResult(), maxResult < MinResult.");
        }
        MaxResult = maxResult;
      }

      template <typename T>
      T Mutagen<T>::GetVariabilityForce() const noexcept
      {
        return VariabilityForce;
      }

      template <typename T>
      void Mutagen<T>::SetVariabilityForce(const T variabilityForce)
      {
        if (variabilityForce < 0)
        {
          throw std::invalid_argument("cnn::engine::common::Mutagen::SetVariabilityForce(), variabilityForce < 0.");
        }
        VariabilityForce = variabilityForce;
      }

      template <typename T>
      void Mutagen<T>::SetSeed(const unsigned int seed) noexcept
      {
        DRE.seed(seed);
      }

      template <typename T>
      void Mutagen<T>::Clear() noexcept
      {
        MinResult = static_cast<T>(0.L);
        MaxResult = static_cast<T>(0.L);
        VariabilityForce = static_cast<T>(0.L);
        DRE.seed(0);
      }

      template <typename T>
      T Mutagen<T>::Mutate(const T value) noexcept
      {
        T result{ value };

        // Variability.
        {
          std::uniform_real_distribution<T> urd{ -VariabilityForce, +VariabilityForce };
          result += urd(DRE);
        }

        // Range control.
        {
          if (result < MinResult)
          {
            return MinResult;
          }
          if (result > MaxResult)
          {
            return MaxResult;
          }
          return result;
        }
      }
    }
  }
}