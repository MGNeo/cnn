#include "LayerTopology.hpp"

namespace cnn
{
  namespace engine
  {
    namespace perceptron
    {

      LayerTopology::LayerTopology(const size_t inputCount,
        const size_t neuronCount)
        :
        InputCount{ inputCount },
        NeuronCount{ neuronCount }
      {
      }

      LayerTopology::LayerTopology(LayerTopology&& topology) noexcept
        :
        InputCount{ topology.InputCount },
        NeuronCount{ topology.NeuronCount }
      {
        topology.Clear();
      }

      LayerTopology& LayerTopology::operator=(LayerTopology&& topology) noexcept
      {
        if (this != &topology)
        {
          InputCount = topology.InputCount;
          NeuronCount = topology.NeuronCount;

          topology.Clear();
        }
        return *this;
      }

      bool LayerTopology::operator==(const LayerTopology& topology) const noexcept
      {
        if ((InputCount == topology.InputCount) && (NeuronCount == topology.NeuronCount))
        {
          return true;
        } else {
          return false;
        }
      }

      bool LayerTopology::operator!=(const LayerTopology& topology) const noexcept
      {
        if (*this == topology)
        {
          return false;
        } else {
          return true;
        }
      }

      size_t LayerTopology::GetInputCount() const noexcept
      {
        return InputCount;
      }

      void LayerTopology::SetInputCount(const size_t inputCount) noexcept
      {
        InputCount = inputCount;
      }

      size_t LayerTopology::GetNeuronCount() const noexcept
      {
        return NeuronCount;
      }

      void LayerTopology::SetNeuronCount(const size_t neuronCount) noexcept
      {
        NeuronCount = neuronCount;
      }

      void LayerTopology::Clear() noexcept
      {
        InputCount = 0;
        NeuronCount = 0;
      }

      void LayerTopology::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::perceptron::LayerTopology::Save(), ostream.good() == false.");
        }

        ostream.write(reinterpret_cast<const char* const>(&InputCount), sizeof(InputCount));
        ostream.write(reinterpret_cast<const char* const>(&NeuronCount), sizeof(NeuronCount));

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::perceptron::LayerTopology::Save(), ostream.good() == false.");
        }
      }

      void LayerTopology::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::perceptron::LayerTopology::Load(), istream.good() == false.");
        }

        decltype(InputCount) inputCount{};
        decltype(NeuronCount) neuronCount{};

        istream.read(reinterpret_cast<char* const>(&inputCount), sizeof(inputCount));
        istream.read(reinterpret_cast<char* const>(&neuronCount), sizeof(neuronCount));

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::perceptron::LayerTopology::Load(), istream.good() == false.");
        }

        InputCount = inputCount;
        NeuronCount = neuronCount;

      }
    }
  }
}