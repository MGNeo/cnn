#pragma once

#include <memory>
#include <type_traits>
#include <stdexcept>

#include "i_map.hpp"

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      template <typename T>
      class Map : public IMap<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Map(const size_t valueCount);

        size_t GetValueCount() const override;

        T GetValue(const size_t index) const override;
        void SetValue(const size_t index, const T value) override;

        void Clear() override;

        typename IMap<T>::Uptr Clone(const bool cloneState) const override;

        Map(const Map<T>& map, const bool cloneState);

      private:

        size_t ValueCount;
        std::unique_ptr<T[]> Values;

      };

      template <typename T>
      Map<T>::Map(const size_t valueCount)
        :
        ValueCount{ valueCount }
      {
        if (ValueCount == 0)
        {
          throw std::invalid_argument("cnn::engine::common::Map::Map(), ValueCount == 0.");
        }
        Values = std::make_unique<T[]>(ValueCount);
        Clear();
      }

      template <typename T>
      size_t Map<T>::GetValueCount() const
      {
        return ValueCount;
      }

      template <typename T>
      T Map<T>::GetValue(const size_t index) const
      {
        if (index >= ValueCount)
        {
          throw std::range_error("cnn::engine::common::Map::GetValue(), index >= ValueCount.");
        }
        return Values[index];
      }
      
      template <typename T>
      void Map<T>::SetValue(const size_t index, const T value)
      {
        if (index >= ValueCount)
        {
          throw std::range_error("cnn::engine::common::Map::SetValue(), index >= ValueCount.");
        }
        Values[index] = value;
      }

      template <typename T>
      void Map<T>::Clear()
      {
        for (size_t i = 0; i < ValueCount; ++i)
        {
          Values[i] = 0;
        }
      }

      template <typename T>
      typename IMap<T>::Uptr Map<T>::Clone(const bool cloneState) const
      {
        return std::make_unique<Map<T>>(*this, cloneState);
      }

      template <typename T>
      Map<T>::Map(const Map<T>& map, const bool cloneState)
        :
        ValueCount{ map.GetValueCount() },
        Values{ std::make_unique<T[]>(ValueCount) }
      {
        if (cloneState == true)
        {
          memcpy(Values.get(), map.Values.get(), sizeof(T) * ValueCount);
        } else {
          for (size_t v = 0; v < ValueCount; ++v)
          {
            Values[v] = 0;
          }
        }
      }
    }
  }
}