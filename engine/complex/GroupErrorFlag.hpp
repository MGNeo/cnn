#pragma once

#include <atomic>
#include <functional>

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      class GroupErrorFlag
      {
      public:

        GroupErrorFlag();

        bool IsError() const noexcept;

        void SetUp() noexcept;

      private:

        std::atomic<bool> Error;

      };
    }
  }
}
