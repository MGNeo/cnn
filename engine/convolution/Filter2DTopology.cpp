#include "Filter2DTopology.hpp"

#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      Filter2DTopology::Filter2DTopology(const Size2D& size, const size_t coreCount) noexcept
        :
        Size{ size },
        CoreCount{ coreCount }
      {
      }

      Filter2DTopology::Filter2DTopology(Filter2DTopology&& topology) noexcept
        :
        Size{ topology.GetSize() },
        CoreCount{ topology.CoreCount }
      {
        topology.Reset();
      }

      Filter2DTopology& Filter2DTopology::operator=(Filter2DTopology&& topology) noexcept
      {
        if (this != &topology)
        {
          Size = std::move(topology.Size);
          CoreCount = topology.CoreCount;

          topology.Reset();
        }
        return *this;
      }

      bool Filter2DTopology::operator==(const Filter2DTopology& topology) const noexcept
      {
        if ((Size == topology.Size) && (CoreCount == topology.CoreCount))
        {
          return true;
        } else {
          return false;
        }
      }

      bool Filter2DTopology::operator!=(const Filter2DTopology& topology) const noexcept
      {
        if (*this == topology)
        {
          return false;
        } else {
          return true;
        }
      }

      const Size2D& Filter2DTopology::GetSize() const noexcept
      {
        return Size;
      }

      void Filter2DTopology::SetSize(const Size2D& size) noexcept
      {
        Size = size;
      }

      size_t Filter2DTopology::GetCoreCount() const noexcept
      {
        return CoreCount;
      }

      void Filter2DTopology::SetCoreCount(const size_t coreCount) noexcept
      {
        CoreCount = coreCount;
      }

      void Filter2DTopology::Reset() noexcept
      {
        Size.Reset();
        CoreCount = 0;
      }

      void Filter2DTopology::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Filter2DTopology::Save(), ostream.good() == false.");
        }
        Size.Save(ostream);
        ostream.write(reinterpret_cast<const char* const>(&CoreCount), sizeof(CoreCount));
        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Filter2DTopology::Save(), ostream.good() == false.");
        }
      }

      void Filter2DTopology::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Filter2DTopology::Load(), istream.good() == false.");
        }

        decltype(Size) size;
        decltype(CoreCount) coreCount{};

        size.Load(istream);
        istream.read(reinterpret_cast<char* const>(&coreCount), sizeof(coreCount));

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Filter2DTopology::Load(), istream.good() == false.");
        }

        Size = std::move(size);
        CoreCount = coreCount;
      }
    }
  }
}