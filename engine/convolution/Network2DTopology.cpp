#include "Network2DTopology.hpp"

#include <stdexcept>

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      Network2DTopology::Network2DTopology()
      {
        Topologies.reserve(16);
      }

      Network2DTopology& Network2DTopology::operator=(const Network2DTopology& topology)
      {
        if (this != &topology)
        {
          Network2DTopology tmpTopology{ topology };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpTopology);
        }
        return *this;
      }

      bool Network2DTopology::operator==(const Network2DTopology& topology) const noexcept
      {
        if (Topologies.size() != topology.Topologies.size())
        {
          return false;
        }

        for (size_t i = 0; i < Topologies.size(); ++i)
        {
          if (Topologies[i] != topology.Topologies[i])
          {
            return false;
          }
        }

        return true;
      }

      bool Network2DTopology::operator!=(const Network2DTopology& topology) const noexcept
      {
        if (*this == topology)
        {
          return false;
        } else {
          return true;
        }
      }

      void Network2DTopology::PushBack(const Layer2DTopology& topology)
      {
        Topologies.push_back(topology);
      }

      size_t Network2DTopology::GetLayerCount() const noexcept
      {
        return Topologies.size();
      }

      const Layer2DTopology& Network2DTopology::GetLayerTopology(const size_t index) const
      {
        if (index >= Topologies.size())
        {
          throw std::range_error("cnn::engine::convolution::Network2DTopology::GetLayerTopology(), index >= Topologies.size().");
        }
        return Topologies[index];
      }

      const Layer2DTopology& Network2DTopology::GetFirstLayerTopology() const
      {
        if (Topologies.size() == 0)
        {
          throw std::logic_error("cnn::engine::convolution::Network2DTopology::GetFirstLayerTopology(), Topologies.size() == 0.");
        }
        return Topologies.front();
      }

      const Layer2DTopology& Network2DTopology::GetLastLayerTopology() const
      {
        if (Topologies.size() == 0)
        {
          throw std::logic_error("cnn::engine::convolution::Network2DTopology::GetLastLayerTopology(), Topologies.size() == 0.");
        }
        return Topologies.back();
      }

      void Network2DTopology::Reset() noexcept
      {
        Topologies.clear();
      }

      void Network2DTopology::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Network2DTopology::Save(), ostream.good() == false.");
        }

        const size_t count = Topologies.size();
        ostream.write(reinterpret_cast<const char*const>(&count), sizeof(count));

        for (size_t i = 0; i < count; ++i)
        {
          Topologies[i].Save(ostream);
        }

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Network2DTopology::Save(), ostream.good() == false.");
        }
      }

      void Network2DTopology::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Network2DTopology::Load(), istream.good() == false.");
        }

        size_t count{};

        decltype(Topologies) topologies;
        topologies.reserve(16);
        
        istream.read(reinterpret_cast<char* const>(&count), sizeof(count));

        topologies.resize(count);

        for (size_t i = 0; i < count; ++i)
        {
          topologies[i].Load(istream);
        }

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Network2DTopology::Load(), istream.good() == false.");
        }

        Topologies = std::move(topologies);
      }

    }
  }
}