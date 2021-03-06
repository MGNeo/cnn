#pragma once

#include <memory>
#include <type_traits>
#include <stdexcept>
#include <cstdint>
#include <cstddef>
#include <istream>
#include <ostream>
#include <cstring>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class Map
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Map(const size_t valueCount = 0);

        Map(const Map& map);

        Map(Map&& map) noexcept;

        // Exception guarantee: strong for this.
        Map& operator=(const Map& map);

        Map& operator=(Map&& map) noexcept;

        size_t GetValueCount() const noexcept;

        // Exception guarantee: strong for this.
        void SetValueCount(const size_t valueCount);

        // Exception guarantee: strong for this.
        T GetValue(const size_t index) const;

        // Exception guarantee: strong for this.
        void SetValue(const size_t index, const T value);

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

        // Exception guarantee: strong for this.
        // Topologies of this and map must be equal.
        void FillFrom(const Map& map);

      private:

        size_t ValueCount;
        std::unique_ptr<T[]> Values;

      };

      template <typename T>
      Map<T>::Map(const size_t valueCount)
        :
        ValueCount{ valueCount }
      {
        Values = std::make_unique<T[]>(ValueCount);
        Clear();
      }

      template <typename T>
      Map<T>::Map(const Map& map)
        :
        ValueCount{ map.ValueCount }
      {
        Values = std::make_unique<T[]>(ValueCount);
        std::memcpy(Values.get(), map.Values.get(), sizeof(T) * ValueCount);
      }

      template <typename T>
      Map<T>::Map(Map&& map) noexcept
        :
        ValueCount{ map.ValueCount },
        Values{ std::move(map.Values) }
      {
        map.Reset();
      }


      template <typename T>
      Map<T>& Map<T>::operator=(const Map& map)
      {
        if (this != &map)
        {
          Map tmpMap{ map };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpMap);
        }
        return *this;
      }

      template <typename T>
      Map<T>& Map<T>::operator=(Map&& map) noexcept
      {
        if (this != &map)
        {
          ValueCount = map.ValueCount;
          Values = std::move(map.Values);

          map.Reset();
        }
        return *this;
      }

      template <typename T>
      size_t Map<T>::GetValueCount() const noexcept
      {
        return ValueCount;
      }

      template <typename T>
      void Map<T>::SetValueCount(const size_t valueCount)
      {
        Map tmpMap{ valueCount };
        // Beware, it is very intimate place for strong exception guarantee.
        std::swap(*this, tmpMap);
      }

      template <typename T>
      T Map<T>::GetValue(const size_t index) const
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= ValueCount)
        {
          throw std::range_error("cnn::engine::common::Map::GetValue(), index >= ValueCount.");
        }
#endif
        return Values[index];
      }

      template <typename T>
      void Map<T>::SetValue(const size_t index, const T value)
      {
#ifndef CNN_DISABLE_RANGE_CHECKS
        if (index >= ValueCount)
        {
          throw std::range_error("cnn::engine::common::Map::SetValue(), index >= ValueCount.");
        }
#endif
        Values[index] = value;
      }

      template <typename T>
      void Map<T>::Clear() noexcept
      {
        for (size_t i = 0; i < ValueCount; ++i)
        {
          Values[i] = static_cast<T>(0.L);
        }
      }

      template <typename T>
      void Map<T>::Reset() noexcept
      {
        ValueCount = 0;
        Values.reset(nullptr);
      }

      template <typename T>
      void Map<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::common::Map::Save(), ostream.good == false.");
        }

        ostream.write(reinterpret_cast<const char*const>(&ValueCount), sizeof(ValueCount));
        for (size_t i = 0; i < ValueCount; ++i)
        {
          ostream.write(reinterpret_cast<const char*const>(&(Values[i])), sizeof(Values[i]));
        }

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::common::Map::Save(), ostream.good == false.");
        }
      }

      template <typename T>
      void Map<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::common::Map::Load(), istream.good == false.");
        }

        decltype(ValueCount) valueCount{};
        decltype(Values) values;

        istream.read(reinterpret_cast<char* const>(&valueCount), sizeof(valueCount));

        values = std::make_unique<T[]>(valueCount);
        for (size_t i = 0; i < valueCount; ++i)
        {
          istream.read(reinterpret_cast<char* const>(&(values[i])), sizeof(values[i]));
        }

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::common::Map::Load(), istream.good == false.");
        }

        ValueCount = valueCount;
        Values = std::move(values);
      }

      template <typename T>
      void Map<T>::FillFrom(const Map& map)
      {
        if (this != &map)
        {
          if (ValueCount != map.ValueCount)
          {
            throw std::invalid_argument("cnn::engine::common::Map::FillFrom(), ValueCount != map.ValueCount.");
          }
          std::memcpy(Values.get(), map.Values.get(), sizeof(T) * ValueCount);
        }
      }
    }
  }
}
