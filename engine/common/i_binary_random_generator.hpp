#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>

namespace cnn
{
  namespace engine
  {
    namespace common
    {
      class IBinaryRandomGenerator
      {
      public:

        using Uptr = std::unique_ptr<IBinaryRandomGenerator>;

        virtual bool Generate() = 0;

        virtual ~IBinaryRandomGenerator() = default;

      };
    }
  }
}