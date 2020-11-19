#pragma once

#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <istream>
#include <ostream>
#include <stdexcept>

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

        Filter2DTopology(const Filter2DTopology& topology) = default;

        Filter2DTopology(Filter2DTopology&& topology) noexcept;

        Filter2DTopology& operator=(const Filter2DTopology& topology) = default;
        
        Filter2DTopology& operator=(Filter2DTopology&& topology) noexcept;

        T GetWidth() const noexcept;

        void SetWidth(const T width) noexcept;

        T GetHeight() const noexcept;

        void SetHeight(const T height) noexcept;

        T GetCoreCount() const noexcept;

        void SetCoreCount(const T coreCount) noexcept;

        void Clear() noexcept;

        // Exception guarantee: base for ostream.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        void Load(std::istream& istream);

      private:

        T Width;
        T Height;
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
        Width{ topology.Width },
        Height{ topology.Height },
        CoreCount{ topology.CoreCount }
      {
      }

      template <typename T>
      Filter2DTopology<T>& Filter2DTopology<T>::operator=(Filter2DTopology&& topology) noexcept
      {
        if (this != &topology)
        {
          Width = topology.Width;
          Height = topology.Height;
          CoreCount = topology.CoreCount;

          topology.Clear();
        }
        return *this;
      }

      template <typename T>
      T Filter2DTopology<T>::GetWidth() const noexcept
      {
        return Width;
      }

      template <typename T>
      void Filter2DTopology<T>::SetWidth(const T width) noexcept
      {
        Width = width;
      }

      template <typename T>
      T Filter2DTopology<T>::GetHeight() const noexcept
      {
        return Height;
      }

      template <typename T>
      void Filter2DTopology<T>::SetHeight(const T height) noexcept
      {
        Height = height;
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
        Width = 0;
        Height = 0;
        CoreCount = 0;
      }

      template <typename T>
      void Filter2DTopology<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Filter2DTopology::Save(), ostream.good() == false.");
        }
        ostream.write(reinterpret_cast<const char*const>(&Width), sizeof(Width));
        ostream.write(reinterpret_cast<const char*const>(&Height), sizeof(Height));
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

        decltype(Width) width{};
        decltype(Height) height{};
        decltype(CoreCount) coreCount{};

        istream.read(reinterpret_cast<char* const>(&width), sizeof(width));
        istream.read(reinterpret_cast<char* const>(&height), sizeof(height));
        istream.read(reinterpret_cast<char* const>(&coreCount), sizeof(coreCount));

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Filter2DTopology::Load(), istream.good() == false.");
        }

        SetWidth(width);
        SetHeight(height);
        SetCoreCount(coreCount);
      }
    }
  }
}