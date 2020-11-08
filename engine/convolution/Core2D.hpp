#pragma once

#include <type_traits>
#include <istream>
#include <ostream>

#include "../common/Neuron.hpp"
#include "../common/Size2D.hpp"

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

        Core2D(const common::Size2D<size_t> size = {});

        Core2D(const Core2D& core) = default;

        Core2D(Core2D&& core) noexcept;

        Core2D& operator=(const Core2D& core) = default;

        Core2D& operator=(Core2D&& core) noexcept;

        common::Size2D<size_t> GetSize() const noexcept;

        void SetSize(const common::Size2D<size_t> size);

        T GetInput(const size_t x, const size_t y) const;

        void SetInput(const size_t x, const size_t y, const T value);

        T GetWeight(const size_t x, const size_t y) const;

        void SetWeight(const size_t x, const size_t y, const T value);

        void GenerateOutput();

        T GetOutput() const;

        // It clears the state without changing of the topology.
        void ClearInputs();

        // It clears the state without changing of the topology.
        void ClearWeights();

        // It clears the state without changing of the topology.
        void ClearOutput();

        // It clears the state without changing of the topology.
        void Clear();

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

        common::Size2D<size_t> Size;

        common::Neuron<T> Neuron;

        size_t ToIndex(const size_t x, const size_t y) const;

      };

      template <typename T>
      Core2D<T>::Core2D(const common::Size2D<size_t> size)
      {
        Size = size;
        Neuron.SetInputCount(Size.GetArea());

        Clear();
      }

      template <typename T>
      Core2D<T>::Core2D(Core2D&& core) noexcept
        :
        Size{ std::move(core.Size) },
        Neuron{ std::move(core.Neuron) }
      {
        core.Clear();
      }

      template <typename T>
      Core2D<T>& Core2D<T>::operator=(Core2D&& core) noexcept
      {
        if (this != &core)
        {
          Size = std::move(core.Size);
          Neuron = std::move(core.Neuron);
        }
        return *this;
      }

      template <typename T>
      common::Size2D<size_t> Core2D<T>::GetSize() const noexcept
      {
        return Size;
      }

      template <typename T>
      void Core2D<T>::SetSize(const common::Size2D<size_t> size)
      {
        Core2D<T> core{ size };
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
      void Core2D<T>::GenerateOutput()
      {
        Neuron.GenerateOutput();
      }

      template <typename T>
      T Core2D<T>::GetOutput() const
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
      void Core2D<T>::ClearInputs()
      {
        Neuron.ClearInputs();
      }

      template <typename T>
      void Core2D<T>::ClearWeights()
      {
        Neuron.ClearWeights();
      }

      template <typename T>
      void Core2D<T>::ClearOutput()
      {
        Neuron.ClearOutput();
      }

      template <typename T>
      void Core2D<T>::Clear()
      {
        Neuron.Clear();
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

      // ??????????
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
          throw std::logic_error("cnn::engine::convolution::Core2D::Load(), neuron != size.GetArea().");
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