#include "Layer2DTopology.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      Layer2DTopology::Layer2DTopology(const Size2D inputSize,
                                       const size_t inputCount,
                                       const Filter2DTopology filterTopology,
                                       const size_t filterCount,
                                       const Size2D outputSize,
                                       const size_t outputCount)
        :
        InputSize{ inputSize },
        InputCount{ inputCount },
        FilterTopology{ filterTopology },
        FilterCount{ filterCount },
        OutputSize{ outputSize },
        OutputCount{ outputCount }
      {
      }

      Layer2DTopology::Layer2DTopology(Layer2DTopology&& topology) noexcept
        :
        InputSize{ topology.InputSize },
        InputCount{ topology.InputCount },
        FilterTopology{ topology.FilterTopology },
        FilterCount{ topology.FilterCount },
        OutputSize{ topology.OutputSize },
        OutputCount{ topology.OutputCount }
      {
        topology.Clear();
      }

      Layer2DTopology& Layer2DTopology::operator=(Layer2DTopology&& topology) noexcept
      {
        if (this != &topology)
        {
          InputSize = topology.InputSize;
          InputCount = topology.InputCount;
          FilterTopology = topology.FilterTopology;
          FilterCount = topology.FilterCount;
          OutputSize = topology.OutputSize;
          OutputCount = topology.OutputCount;

          topology.Clear();
        }
        return *this;
      }

      bool Layer2DTopology::operator==(const Layer2DTopology& topology) const
      {
        if ((InputSize == topology.InputSize) &&
            (InputCount == topology.InputCount) &&
            (FilterTopology == topology.FilterTopology) &&
            (FilterCount == topology.FilterCount) &&
            (OutputSize == topology.OutputSize) &&
            (OutputCount == topology.OutputCount))
        {
          return true;
        } else {
          return false;
        }
      }

      bool Layer2DTopology::operator!=(const Layer2DTopology& topology) const
      {
        if (*this == topology)
        {
          return false;
        } else {
          return true;
        }
      }

      Size2D Layer2DTopology::GetInputSize() const noexcept
      {
        return InputSize;
      }

      void Layer2DTopology::SetInputSize(const Size2D& inputSize)
      {
        InputSize = inputSize;
      }

      size_t Layer2DTopology::GetInputCount() const noexcept
      {
        return InputCount;
      }

      void Layer2DTopology::SetInputCount(const size_t inputCount)
      {
        InputCount = inputCount;
      }

      Filter2DTopology Layer2DTopology::GetFilterTopology() const noexcept
      {
        return FilterTopology;
      }

      void Layer2DTopology::SetFilterTopology(const Filter2DTopology& filterTopology) noexcept
      {
        FilterTopology = filterTopology;
      }

      size_t Layer2DTopology::GetFilterCount() const noexcept
      {
        return FilterCount;
      }

      void Layer2DTopology::SetFilterCount(const size_t filterCount) noexcept
      {
        FilterCount = filterCount;
      }

      Size2D Layer2DTopology::GetOutputSize() const noexcept
      {
        return OutputSize;
      }

      void Layer2DTopology::SetOutputSize(const Size2D& outputSize) noexcept
      {
        OutputSize = outputSize;
      }

      size_t Layer2DTopology::GetOutputCount() const noexcept
      {
        return OutputCount;
      }

      void Layer2DTopology::SetOutputCount(const size_t outputCount) noexcept
      {
        OutputCount = outputCount;
      }

      void Layer2DTopology::Clear() noexcept
      {
        InputSize.Clear();
        InputCount = 0;
        FilterTopology.Clear();
        FilterCount = 0;
        OutputSize.Clear();
        OutputCount = 0;
      }

      void Layer2DTopology::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2DTopology::Save(), ostream.good() == false.");
        }

        InputSize.Save(ostream);
        ostream.write(reinterpret_cast<const char*const>(&InputCount), sizeof(InputCount));
        FilterTopology.Save(ostream);
        ostream.write(reinterpret_cast<const char* const>(&FilterCount), sizeof(FilterCount));
        OutputSize.Save(ostream);
        ostream.write(reinterpret_cast<const char* const>(&OutputCount), sizeof(OutputCount));

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Layer2DTopology::Save(), ostream.good() == false.");
        }
      }

      void Layer2DTopology::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2DTopology::Load(), istream.good() == false.");
        }

        decltype(InputSize) inputSize;
        decltype(InputCount) inputCount{};
        decltype(FilterTopology) filterTopology;
        decltype(FilterCount) filterCount{};
        decltype(OutputSize) outputSize;
        decltype(OutputCount) outputCount{};

        inputSize.Load(istream);
        istream.read(reinterpret_cast<char*const>(&inputCount), sizeof(inputCount));
        filterTopology.Load(istream);
        istream.read(reinterpret_cast<char*const>(&filterCount), sizeof(filterCount));
        outputSize.Load(istream);
        istream.read(reinterpret_cast<char*const>(&outputCount), sizeof(outputCount));

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Layer2DTopology::Load(), istream.good() == false.");
        }

        InputSize = std::move(inputSize);
        InputCount = inputCount;
        FilterTopology = std::move(filterTopology);
        FilterCount = filterCount;
        OutputSize = std::move(outputSize);
        OutputCount = outputCount;
      }
    }
  }
}