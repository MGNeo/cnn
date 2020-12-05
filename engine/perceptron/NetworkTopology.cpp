#include "NetworkTopology.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {
      NetworkTopology::NetworkTopology()
      {
        Topologies.reserve(16);
      }

      NetworkTopology& NetworkTopology::operator=(const NetworkTopology& topology)
      {
        if (this != &topology)
        {
          NetworkTopology tmpTopology{ topology };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpTopology);
        }
        return *this;
      }

      bool NetworkTopology::operator==(const NetworkTopology& topology) const noexcept
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

      bool NetworkTopology::operator!=(const NetworkTopology& topology) const noexcept
      {
        if (*this == topology)
        {
          return false;
        } else {
          return true;
        }
      }

      void NetworkTopology::PushBack(const LayerTopology& topology)
      {
        Topologies.push_back(topology);
      }

      size_t NetworkTopology::GetLayerCount() const noexcept
      {
        return Topologies.size();
      }

      LayerTopology NetworkTopology::GetLayerTopology(const size_t index) const
      {
        if (index >= Topologies.size())
        {
          throw std::range_error("cnn::engine::perceptron::NetworkTopology::GetLayerTopology(), index >= Topologies.size().");
        }
        return Topologies[index];
      }

      LayerTopology NetworkTopology::GetFirstLayerTopology() const
      {
        if (Topologies.size() == 0)
        {
          throw std::logic_error("cnn::engine::perceptron::NetworkTopology::GetFirstLayerTopology(), Topologies.size() == 0.");
        }
        return Topologies.front();
      }

      LayerTopology NetworkTopology::GetLastLayerTopology() const
      {
        if (Topologies.size() == 0)
        {
          throw std::logic_error("cnn::engine::perceptron::NetworkTopology::GetLastLayerTopology(), Topologies.size() == 0.");
        }
        return Topologies.back();
      }

      void NetworkTopology::Reset() noexcept
      {
        Topologies.clear();
      }

      void NetworkTopology::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::perceptron::NetworkTopology::Save(), ostream.good() == false.");
        }

        const size_t count = Topologies.size();
        ostream.write(reinterpret_cast<const char*const>(&count), sizeof(count));

        for (size_t i = 0; i < count; ++i)
        {
          Topologies[i].Save(ostream);
        }

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::perceptron::NetworkTopology::Save(), ostream.good() == false.");
        }
      }

      void NetworkTopology::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::perceptron::NetworkTopology::Load(), istream.good() == false.");
        }

        size_t count{};
        decltype(Topologies) topologies;

        istream.read(reinterpret_cast<char* const>(&count), sizeof(count));

        topologies.resize(count);
        for (size_t i = 0; i < topologies.size(); ++i)
        {
          topologies[i].Load(istream);
        }

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::perceptron::NetworkTopology::Load(), istream.good() == false.");
        }

        Topologies = std::move(topologies);

      }
    }
  }
}