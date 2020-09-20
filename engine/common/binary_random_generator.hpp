#pragma once

#include "i_binary_random_generator.hpp"

#include <random>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      class BinaryRandomGenerator : public IBinaryRandomGenerator
      {
      public:

        using Uptr = std::unique_ptr<BinaryRandomGenerator>;

        inline BinaryRandomGenerator(const size_t seed = 0);

        inline bool Generate() override;

      private:

        std::default_random_engine DRE;
        std::uniform_int_distribution<uint64_t> UID;

        size_t Pos;
        uint64_t Values;

      };

      BinaryRandomGenerator::BinaryRandomGenerator(const size_t seed)
        :
        DRE{ static_cast<size_t>(time(NULL)) + static_cast<size_t>(clock()) + seed },
        UID{ 0, UINT64_MAX },
        Pos{ 1 },
        Values{}
      {
      }

      bool BinaryRandomGenerator::Generate()
      {
        if (Pos == 64)
        {
          Values = UID(DRE);
          Pos = 0;
        }

        return Values & (1ui64 << (Pos++));
      }
    }
  }
}