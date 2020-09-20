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

        BinaryRandomGenerator(const size_t seed = 0);

        bool Generate() override;

      public:

        std::default_random_engine DRE;
        std::uniform_int_distribution<uint64_t> UID;

        size_t Pos;
        uint64_t Values;

      };
    }
  }
}