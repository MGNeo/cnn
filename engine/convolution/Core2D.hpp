#pragma once

#include <type_traits>
#include <istream>
#include <ostream>

#include "../common/Neuron.hpp"
#include "Size2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Core2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Core2D(const Size2D& size = {});

        Core2D(const Core2D& core) = default;

        Core2D(Core2D&& core) noexcept = default;

        // Exception guarantee: strong for this.
        Core2D& operator=(const Core2D& core);

        Core2D& operator=(Core2D&& core) noexcept = default;

        const Size2D& GetSize() const noexcept;

        // Exception guarantee: strong for this.
        void SetSize(const Size2D& size);

        T GetInput(const size_t x, const size_t y) const;

        // Exception guarantee: strong for this.
        void SetInput(const size_t x, const size_t y, const T value);

        T GetWeight(const size_t x, const size_t y) const;

        // Exception guarantee: strong for this.
        void SetWeight(const size_t x, const size_t y, const T value);

        void GenerateOutput() noexcept;

        T GetOutput() const noexcept;

        // It clears the state without changing of the topology.
        void ClearInputs() noexcept;

        // It clears the state without changing of the topology.
        void ClearWeights() noexcept;

        // It clears the state without changing of the topology.
        void ClearOutput() noexcept;

        // It clears the state without changing of the topology.
        void Clear() noexcept;

        // It resets the state to zero including the topology.
        void Reset() noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        // It loads full state.
        void Load(std::istream& istream);

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) noexcept;

      private:

        Size2D Size;

        common::Neuron<T> Neuron;

        size_t ToIndex(const size_t x, const size_t y) const;

      };

      template <typename T>
      Core2D<T>::Core2D(const Size2D& size)
        :
        Size{ size }, 
        Neuron{ Size.GetArea() }
      {
        Clear();
      }

      template <typename T>
      Core2D<T>& Core2D<T>::operator=(const Core2D<T>& core)
      {
        if (this != &core)
        {
          Core2D<T> tmpCore{ core };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpCore);
        }
        return *this;
      }

      template <typename T>
      const Size2D& Core2D<T>::GetSize() const noexcept
      {
        return Size;
      }

      template <typename T>
      void Core2D<T>::SetSize(const Size2D& size)
      {
        Core2D<T> core{ size };
        // Beware, it is very intimate place for strong exception guarantee.
        std::swap(core, *this);
      }

      template <typename T>
      T Core2D<T>::GetInput(const size_t x, const size_t y) const
      {
        const size_t index = ToIndex(x, y);
        return Neuron.GetInput(index);
      }

      template <typename T>
      void Core2D<T>::SetInput(const size_t x, const size_t y, const T value)
      {
        const size_t index = ToIndex(x, y);
        Neuron.SetInput(index, value);
      }

      template <typename T>
      T Core2D<T>::GetWeight(const size_t x, const size_t y) const
      {
        const size_t index = ToIndex(x, y);
        return Neuron.GetWeight(index);
      }

      template <typename T>
      void Core2D<T>::SetWeight(const size_t x, const size_t y, const T value)
      {
        const size_t index = ToIndex(x, y);
        Neuron.SetWeight(index, value);
      }

      template <typename T>
      void Core2D<T>::GenerateOutput() noexcept
      {
        Neuron.GenerateOutput();
      }

      template <typename T>
      T Core2D<T>::GetOutput() const noexcept
      {
        return Neuron.GetOutput();
      }

      template <typename T>
      size_t Core2D<T>::ToIndex(const size_t x, const size_t y) const
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (x >= Size.GetWidth())
        {
          throw std::range_error("cnn::engine::convolution::Core2D::ToIndex(), x >= Size.GetWidth().");
        }
        if (y >= Size.GetHeight())
        {
          throw std::range_error("cnn::engine::convolution::Core2D::ToIndex(), y >= Size.GetHeight().");
        }
#endif
        return x + y * Size.GetWidth();
      }

      template <typename T>
      void Core2D<T>::ClearInputs() noexcept
      {
        Neuron.ClearInputs();
      }

      template <typename T>
      void Core2D<T>::ClearWeights() noexcept
      {
        Neuron.ClearWeights();
      }

      template <typename T>
      void Core2D<T>::ClearOutput() noexcept
      {
        Neuron.ClearOutput();
      }

      template <typename T>
      void Core2D<T>::Clear() noexcept
      {
        Neuron.Clear();
      }

      template <typename T>
      void Core2D<T>::Reset() noexcept
      {
        Size.Clear();
        Neuron.Reset();
      }

      template <typename T>
      void Core2D<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Core2D::Save(), ostream.good() == false.");
        }

        Size.Save(ostream);
        Neuron.Save(ostream);

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Core2D::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Core2D<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Core2D::Load(), istream.good() == false.");
        }

        decltype(Size) size;
        decltype(Neuron) neuron;

        size.Load(istream);
        neuron.Load(istream);

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Core2D::Load(), istream.good() == false.");
        }

        if (neuron.GetInputCount() != size.GetArea())
        {
          throw std::logic_error("cnn::engine::convolution::Core2D::Load(), neuron.GetInputCount() != size.GetArea().");
        }

        Size = std::move(size);
        Neuron = std::move(neuron);
      }

      template <typename T>
      void Core2D<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept
      {
        Neuron.FillWeights(valueGenerator);
      }

      template <typename T>
      void Core2D<T>::Mutate(common::Mutagen<T>& mutagen) noexcept
      {
        Neuron.Mutate(mutagen);
      }
    }
  }
}
