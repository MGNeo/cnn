#include "GroupErrorFlag.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      GroupErrorFlag::GroupErrorFlag()
        :
        Error{ false }
      {
      }

      bool GroupErrorFlag::IsError() const noexcept
      {
        return Error.load();
      }

      void GroupErrorFlag::SetUp() noexcept
      {
        Error.store(true);
      }
    }
  }
}