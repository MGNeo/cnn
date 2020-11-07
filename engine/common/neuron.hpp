#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <memory>
#include <istream>
#include <ostream>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class Neuron
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Neuron(const size_t inputCount = 0);

        Neuron(const Neuron& neuron);

        Neuron(Neuron&& neuron) noexcept;

        // Exception guarantee: strong for this.
        Neuron& operator=(const Neuron& neuron);

        Neuron& operator=(Neuron&& neuron) noexcept;

        size_t GetInputCount() const noexcept;

        // Exception guarantee: strong for this.
        void SetInputCount(const size_t inputCount);

        T GetInput(const size_t index) const;

        // Exception guarantee: strong for this.
        void SetInput(const size_t index, const T value);

        T GetWeight(const size_t index) const;

        // Exception guarantee: strong for this.
        void SetWeight(const size_t index, const T value);

        T GetOutput() const noexcept;

        // Exception guarantee: strong for this.
        void SetOutput(const T value);

        void GenerateOutput() noexcept;

        void Clear() noexcept;

        void ClearInputs() noexcept;

        void ClearWeights() noexcept;

        void ClearOutput() noexcept;

        // Exception guarantee: strong for this and base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream);

        // Exception guarantee: strong for this and base for istream.
        // It loads full state.
        void Load(std::istream& istream);

      private:

        // InputCount % 2 must be zero.
        size_t InputCount;
        
        std::unique_ptr<T[]> Inputs;
        
        std::unique_ptr<T[]> Weights;
        
        T Output;

      };

      template <typename T>
      Neuron<T>::Neuron(const size_t inputCount)
      {
        if ((inputCount % 2) != 0)
        {
          throw std::invalid_argument("cnn::engine::common::Neuron::Neuron(), (inputCount % 2) != 0.");
        }

        InputCount = inputCount;

        Inputs = std::make_unique<T[]>(InputCount);
        Weights = std::make_unique<T[]>(InputCount);

        Clear();
      }

      template <typename T>
      Neuron<T>::Neuron(const Neuron& neuron)
        :
        InputCount{ neuron.InputCount },
        Output{ neuron.Output }
      {
        if (InputCount != 0)
        {
          Inputs = std::make_unique<T[]>(InputCount);
          memcpy(Inputs.get(), neuron.Inputs.get(), sizeof(T) * InputCount);

          Weights = std::make_unique<T[]>(InputCount);
          memcpy(Weights.get(), neuron.Weights.get(), sizeof(T) * InputCount);
        }
      }

      template <typename T>
      Neuron<T>::Neuron(Neuron&& neuron) noexcept
        :
        InputCount{ neuron.InputCount },
        Inputs{ std::move(neuron.Inputs) },
        Weights{ std::move(neuron.Weights) },
        Output{ neuron.Output }
      {
        neuron.InputCount = 0;
        neuron.Output = static_cast<T>(0.L);
      }

      template <typename T>
      Neuron<T>& Neuron<T>::operator=(const Neuron& neuron)
      {
        if (this != &neuron)
        {
          Neuron tmpNeuron{ neuron };
          std::swap(tmpNeuron, *this);
        }
        return *this;
      }

      template <typename T>
      Neuron<T>& Neuron<T>::operator=(Neuron&& neuron) noexcept
      {
        if (this != &neuron)
        {
          InputCount = neuron.InputCount;
          Inputs = std::move(neuron.Inputs);
          Weights = std::move(neuron.Weights);
          Output = neuron.Output;

          neuron.InputCount = 0;
          neuron.Output = static_cast<T>(0.L);
        }
        return *this;
      }

      template <typename T>
      size_t Neuron<T>::GetInputCount() const noexcept
      {
        return InputCount;
      }

      template <typename T>
      void Neuron<T>::SetInputCount(const size_t inputCount)
      {
        if (InputCount != inputCount)
        {
          Neuron neuron(inputCount);
          std::swap(neuron, *this);
        }
      }

      template <typename T>
      T Neuron<T>::GetInput(const size_t index) const
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::GetInput(), index >= InputCount.");
        }
        return Inputs[index];
      }

      template <typename T>
      void Neuron<T>::SetInput(const size_t index, const T value)
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::SetInput(), index >= InputCount.");
        }
        Inputs[index] = value;
      }

      template <typename T>
      T Neuron<T>::GetWeight(const size_t index) const
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::GetWeight(), index >= InputCount.");
        }
        return Weights[index];
      }

      template <typename T>
      void Neuron<T>::SetWeight(const size_t index, const T value)
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::common::Neuron::SetWeight(), index >= InputCount.");
        }
        Weights[index] = value;
      }

      template <typename T>
      T Neuron<T>::GetOutput() const noexcept
      {
        return Output;
      }

      template <typename T>
      void Neuron<T>::SetOutput(const T value)
      {
        Output = value;
      }

      template <typename T>
      void Neuron<T>::GenerateOutput() noexcept
      {
        if (InputCount != 0)
        {
          Output = static_cast<T>(0.L);
          for (size_t i = 0; i < InputCount; ++i)
          {
            Output += Inputs[i] * Weights[i];
          }
          Output = 1 / (1 + exp(-Output));
        }
      }

      template <typename T>
      void Neuron<T>::Clear() noexcept
      {
        ClearInputs();
        ClearWeights();
        ClearOutput();
      }

      template <typename T>
      void Neuron<T>::ClearInputs() noexcept
      {
        for (size_t i = 0; i < InputCount; ++i)
        {
          Inputs[i] = static_cast<T>(0.L);
        }
      }

      template <typename T>
      void Neuron<T>::ClearWeights() noexcept
      {
        for (size_t i = 0; i < InputCount; ++i)
        {
          Weights[i] = static_cast<T>(0.L);
        }
      }

      template <typename T>
      void Neuron<T>::ClearOutput() noexcept
      {
        Output = static_cast<T>(0.L);
      }

      template <typename T>
      void Neuron<T>::Save(std::ostream& ostream)
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::Neuron::Save(), ostream.good() == false.");
        }
        ostream.write(reinterpret_cast<char*const>(&InputCount), sizeof(InputCount));
        for (size_t i = 0; i < InputCount; ++i)
        {
          ostream.write(reinterpret_cast<const char*const>(&(Inputs[i])), sizeof(T));
        }
        for (size_t i = 0; i < InputCount; ++i)
        {
          ostream.write(reinterpret_cast<const char* const>(&(Weights[i])), sizeof(T));
        }
        ostream.write(reinterpret_cast<char* const>(&Output), sizeof(Output));
        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::Neuron::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Neuron<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::Neuron::Load(), istream.good() == false.");
        }
        decltype(InputCount) inputCount{};
        istream.read(reinterpret_cast<char*const>(&inputCount), sizeof(inputCount));
        Neuron tmpNeuron{ inputCount };
        for (size_t i = 0; i < inputCount; ++i)
        {
          T tmpInput{};
          istream.read(reinterpret_cast<char*const>(&tmpInput), sizeof(tmpInput));
          tmpNeuron.SetInput(i, tmpInput);
        }
        for (size_t i = 0; i < inputCount; ++i)
        {
          T tmpWeight{};
          istream.read(reinterpret_cast<char* const>(&tmpWeight), sizeof(tmpWeight));
          tmpNeuron.SetWeight(i, tmpWeight);
        }
        decltype(Output) tmpOutput{};
        istream.read(reinterpret_cast<char* const>(&tmpOutput), sizeof(tmpOutput));
        tmpNeuron.SetOutput(tmpOutput);
        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::Neuron::Load(), istream.good() == false.");
        }
        std::swap(tmpNeuron, *this);
      }
    }
  }
}

