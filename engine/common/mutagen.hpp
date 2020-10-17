#pragma once

#include "i_mutagen.hpp"

#include <random>
#include <stdexcept>
#include <chrono>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class Mutagen : public IMutagen<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<Mutagen<T>>;

        Mutagen();

        T GetMinResult() const override;
        void SetMinResult(const T minResult) override;

        T GetMaxResult() const override;
        void SetMaxResult(const T maxResult) override;
        
        T GetMutationProbability() const override;
        void SetMutationProbability(const T mutationProbability) override;

        T GetMutationForce() const override;
        void SetMutationForce(const T mutationForce) override;

        T GetVariabilityForce() const override;
        void SetVariabilityForce(const T variabilityForce) override;

        T Mutate(const T value) override;

        typename IMutagen<T>::Uptr Clone() const override;

      private:

        T MinResult;
        T MaxResult;

        T MutationProbability;
        T MutationForce;

        T VariabilityForce;

        std::default_random_engine DRE;

      };

      template <typename T>
      Mutagen<T>::Mutagen()
        :
        MinResult{ static_cast<T>(0L) },
        MaxResult{ static_cast<T>(1L) },
        MutationProbability{ static_cast<T>(0.001L) },
        MutationForce{ static_cast<T>(0.1L) },
        VariabilityForce{ static_cast<T>(0.0001L) },
        DRE{ static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()) }
      {
      }

      template <typename T>
      T Mutagen<T>::GetMinResult() const
      {
        return MinResult;
      }

      template <typename T>
      void Mutagen<T>::SetMinResult(const T minResult)
      {
        if (minResult >= MaxResult)
        {
          throw std::invalid_argument("cnn::engine::common::Mutagen::SetMinResult(), minResult >= MaxResult.");
        }
        MinResult = minResult;
      }

      template <typename T>
      T Mutagen<T>::GetMaxResult() const
      {
        return MaxResult;
      }

      template <typename T>
      void Mutagen<T>::SetMaxResult(const T maxResult)
      {
        if (maxResult <= MinResult)
        {
          throw std::invalid_argument("cnn::engine::common::Mutagen::SetMaxResult(), maxResult <= MinResult.");
        }
        MaxResult = maxResult;
      }

      template <typename T>
      T Mutagen<T>::GetMutationProbability() const
      {
        return MutationProbability;
      }

      template <typename T>
      void Mutagen<T>::SetMutationProbability(const T mutationProbability)
      {
        if (mutationProbability < 0)
        {
          throw std::invalid_argument("cnn::engine::common::Mutagen::SetMutationProbability(), mutationProbability < 0.");
        }
        if (mutationProbability > 1)
        {
          throw std::invalid_argument("cnn::engine::common::Mutagen::SetMutationProbability(), mutationProbability > 1.");
        }
        MutationProbability = mutationProbability;
      }

      template <typename T>
      T Mutagen<T>::GetMutationForce() const
      {
        return MutationForce;
      }

      template <typename T>
      void Mutagen<T>::SetMutationForce(const T mutationForce)
      {
        if (mutationForce < 0)
        {
          throw std::invalid_argument("cnn::engine::common::Mutagen::SetMutationForce(), mutationForce < 0.");
        }
        MutationForce = mutationForce;
      }

      template <typename T>
      T Mutagen<T>::GetVariabilityForce() const
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
      T Mutagen<T>::Mutate(const T value)
      {
        T result{ value };

        // Mutation.
        {
          std::uniform_real_distribution<T> urd{ 0, 1 };
          if (urd(DRE) >= MutationProbability)
          {
            std::uniform_real_distribution<T> urd{ -MutationForce, +MutationForce };
            result += urd(DRE);
          }
        }

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

      template <typename T>
      typename IMutagen<T>::Uptr Mutagen<T>::Clone() const
      {
        auto mutagen = std::make_unique<Mutagen<T>>(*this);
        return mutagen;
      }

    }
  }
}