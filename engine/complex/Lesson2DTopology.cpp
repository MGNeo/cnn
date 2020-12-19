#include "Lesson2DTopology.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      Lesson2DTopology::Lesson2DTopology(const convolution::Size2D& inputSize,
                                         const size_t outputCount) noexcept
        :
        InputSize{ inputSize },
        OutputCount{ outputCount }
      {
      }

      Lesson2DTopology::Lesson2DTopology(const Lesson2DTopology& topology) noexcept
        :
        InputSize{ topology.InputSize },
        OutputCount{ topology.OutputCount }
      {
      }

      Lesson2DTopology::Lesson2DTopology(Lesson2DTopology&& topology) noexcept
        :
        InputSize{ std::move(topology.InputSize) },
        OutputCount{ topology.OutputCount }
      {
        topology.Reset();
      }

      Lesson2DTopology& Lesson2DTopology::operator=(const Lesson2DTopology& topology) noexcept
      {
        if (this != &topology)
        {
          InputSize = topology.InputSize;
          OutputCount = topology.OutputCount;
        }
        return *this;
      }

      Lesson2DTopology& Lesson2DTopology::operator=(Lesson2DTopology&& topology) noexcept
      {
        if (this != &topology)
        {
          InputSize = std::move(topology.InputSize);
          OutputCount = std::move(topology.OutputCount);

          topology.Reset();
        }
        return *this;
      }

      bool Lesson2DTopology::operator==(const Lesson2DTopology& topology) const noexcept
      {
        if ((InputSize == topology.InputSize) && (OutputCount == topology.OutputCount))
        {
          return true;
        } else {
          return false;
        }
      }

      bool Lesson2DTopology::operator!=(const Lesson2DTopology& topology) const noexcept
      {
        if (*this == topology)
        {
          return false;
        } else {
          return true;
        }
      }

      const convolution::Size2D& Lesson2DTopology::GetInputSize() const noexcept
      {
        return InputSize;
      }

      void Lesson2DTopology::SetInputSize(const convolution::Size2D& inputSize) noexcept
      {
        InputSize = inputSize;
      }

      size_t Lesson2DTopology::GetOutputCount() const noexcept
      {
        return OutputCount;
      }

      void Lesson2DTopology::SetOutputCount(const size_t outputCount) noexcept
      {
        OutputCount = outputCount;
      }

      void Lesson2DTopology::Reset() noexcept
      {
        InputSize.Reset();
        OutputCount = 0;
      }

      void Lesson2DTopology::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::complex::Lesson2DTopology::Save(), ostream.good() == false.");
        }

        InputSize.Save(ostream);
        ostream.write(reinterpret_cast<const char* const>(&OutputCount), sizeof(OutputCount));

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::complex::Lesson2DTopology::Save(), ostream.good() == false.");
        }
      }

      void Lesson2DTopology::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::complex::Lesson2DTopology::Load(), istream.good() == false.");
        }

        decltype(InputSize) inputSize{};
        decltype(OutputCount) outputCount{};

        istream.read(reinterpret_cast<char* const>(&inputSize), sizeof(inputSize));
        istream.read(reinterpret_cast<char*const>(&outputCount), sizeof(outputCount));

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::complex::Lesson2DTopology::Load(), istream.good() == false.");
        }

        InputSize = inputSize;
        OutputCount = outputCount;

      }
    }
  }
}