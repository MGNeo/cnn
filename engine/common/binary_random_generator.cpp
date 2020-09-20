#include "binary_random_generator.hpp"

#include <time.h>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      BinaryRandomGenerator::BinaryRandomGenerator(const size_t seed)
        :
        DRE{ static_cast<size_t>(time(NULL)) + static_cast<size_t>(clock()) + seed },
        UID{ 0, UINT64_MAX },
        Pos{1},
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