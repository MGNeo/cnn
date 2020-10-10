#pragma once

#include "i_binary_random_generator.hpp"

#include <random>
#include <time.h>

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

        // Inline is necessary for keeping the library as header-only-library.

        inline BinaryRandomGenerator(const unsigned int seed = 0);

        inline bool Generate() override;

      private:

        std::default_random_engine DRE;
        std::uniform_int_distribution<uint64_t> UID;

        size_t Pos;
        uint64_t Values;

      };

      BinaryRandomGenerator::BinaryRandomGenerator(const unsigned int seed)
        :
        DRE{ static_cast<unsigned int>(time(NULL)) + static_cast<unsigned int>(clock()) + seed },
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