#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <istream>
#include <ostream>
#include <stdexcept>

#include "Size2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Filter2DTopology
      {

        static_assert(std::is_integral<T>::value&& std::is_unsigned<T>::value);

      public:

        Filter2DTopology() noexcept;

        Filter2DTopology(const Filter2DTopology& topology) noexcept = default;

        Filter2DTopology(Filter2DTopology&& topology) noexcept;

        Filter2DTopology& operator=(const Filter2DTopology& topology) noexcept = default;
        
        Filter2DTopology& operator=(Filter2DTopology&& topology) noexcept;

        Size2D<T> GetSize() const noexcept;

        void SetSize(const Size2D<size_t>& size) noexcept;

        T GetCoreCount() const noexcept;

        void SetCoreCount(const T coreCount) noexcept;

        void Clear() noexcept;

        // Exception guarantee: base for ostream.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        void Load(std::istream& istream);

      private:

        Size2D<T> Size;
        T CoreCount;

      };

      template <typename T>
      Filter2DTopology<T>::Filter2DTopology() noexcept
      {
        Clear();
      }

      template <typename T>
      Filter2DTopology<T>::Filter2DTopology(Filter2DTopology&& topology) noexcept
        :
        Size{ std::move(topology.Size()) },
        CoreCount{ topology.CoreCount }
      {
        topology.Clear();
      }

      template <typename T>
      Filter2DTopology<T>& Filter2DTopology<T>::operator=(Filter2DTopology&& topology) noexcept
      {
        if (this != &topology)
        {
          Size = std::move(topology.Size);
          CoreCount = topology.CoreCount;

          topology.Clear();
        }
        return *this;
      }

      template <typename T>
      Size2D<T> Filter2DTopology<T>::GetSize() const noexcept
      {
        return Size;
      }

      template <typename T>
      void Filter2DTopology<T>::SetSize(const Size2D<size_t>& size) noexcept
      {
        Size = size;
      }

      template <typename T>
      T Filter2DTopology<T>::GetCoreCount() const noexcept
      {
        return CoreCount;
      }

      template <typename T>
      void Filter2DTopology<T>::SetCoreCount(const T coreCount) noexcept
      {
        CoreCount = coreCount;
      }

      template <typename T>
      void Filter2DTopology<T>::Clear() noexcept
      {
        Size.Clear();
        CoreCount = 0;
      }

      template <typename T>
      void Filter2DTopology<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Filter2DTopology::Save(), ostream.good() == false.");
        }
        Size.Save(ostream);
        ostream.write(reinterpret_cast<const char*const>(&CoreCount), sizeof(CoreCount));
        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Filter2DTopology::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Filter2DTopology<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Filter2DTopology::Load(), istream.good() == false.");
        }

        decltype(Size) size;
        decltype(CoreCount) coreCount{};

        Size.Load(istream);
        istream.read(reinterpret_cast<char* const>(&coreCount), sizeof(coreCount));

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Filter2DTopology::Load(), istream.good() == false.");
        }

        Size = size;
        CoreCount = coreCount;
      }
    }
  }
}