#pragma once

#include "../complex/Lesson2DTopology.hpp"

#include "../convolution/Map2D.hpp"
#include "../convolution/Map2DProtectingReference.hpp"

#include "../common/Map.hpp"
#include "../common/MapProtectingReference.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class Lesson2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Lesson2D(const Lesson2DTopology& topology = {});

        Lesson2D(const Lesson2D& lesson);

        Lesson2D(Lesson2D&& lesson) noexcept = default;

        // Exception guarantee: strong for this.
        Lesson2D& operator=(const Lesson2D& lesson);

        Lesson2D& operator=(Lesson2D&& lesson) noexcept = default;

        const Lesson2DTopology& GetTopology() const noexcept;

        // Exception guarantee: strong for this.
        void SetTopology(const Lesson2DTopology& topology);

        const convolution::Map2D<T>& GetInput() const noexcept;

        convolution::Map2DProtectingReference<T> GetInput() noexcept;

        const common::Map<T>& GetOutput() const noexcept;

        common::MapProtectingReference<T> GetOutput() noexcept;

        // It clears the state without changing of the topology.
        void Clear() noexcept;

        // It resets the state to zero including the topology.
        void Reset() noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        // It loads full state.
        void Load(std::istream& istream);

      private:

        Lesson2DTopology Topology;
        convolution::Map2D<T> Input;
        common::Map<T> Output;

        void CheckTopology(const Lesson2DTopology& topology) const;

      };

      template <typename T>
      Lesson2D<T>::Lesson2D(const Lesson2DTopology& topology)
      {
        CheckTopology(topology);

        Topology = topology;
        Input.SetSize(Topology.GetInputSize());
        Output.SetValueCount(Topology.GetOutputCount());
      }

      template <typename T>
      Lesson2D<T>::Lesson2D(const Lesson2D& lesson)
        :
        Topology{ lesson.Topology }
      {
        Input.SetSize(Topology.GetInputSize());
        Output.SetValueCount(Topology.GetOutputCount());
      }

      template <typename T>
      Lesson2D<T>& Lesson2D<T>::operator=(const Lesson2D& lesson)
      {
        if (this != &lesson)
        {
          Lesson2D<T> tmpLesson{ lesson };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpLesson);
        }
        return *this;
      }

      template <typename T>
      const Lesson2DTopology& Lesson2D<T>::GetTopology() const noexcept
      {
        return Topology;
      }

      template <typename T>
      void Lesson2D<T>::SetTopology(const Lesson2DTopology& topology)
      {
        CheckTopology(topology);

        Lesson2D<T> tmpLesson{ topology };
        // Beware, it is very intimate place for strong exception guarantee.
        std::swap(*this, tmpLesson);
      }

      template <typename T>
      const convolution::Map2D<T>& Lesson2D<T>::GetInput() const noexcept
      {
        return Input;
      }

      template <typename T>
      convolution::Map2DProtectingReference<T> Lesson2D<T>::GetInput() noexcept
      {
        return Input;
      }

      template <typename T>
      const common::Map<T>& Lesson2D<T>::GetOutput() const noexcept
      {
        return Output;
      }

      template <typename T>
      common::MapProtectingReference<T> Lesson2D<T>::GetOutput() noexcept
      {
        return Output;
      }

      template <typename T>
      void Lesson2D<T>::Clear() noexcept
      {
        Input.Clear();
        Output.Clear();
      }

      template <typename T>
      void Lesson2D<T>::Reset() noexcept
      {
        Topology.Reset();
        Input.Reset();
        Output.Reset();
      }

      template <typename T>
      void Lesson2D<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::complex::Lesson2D::Save(), ostream.good() == false.");
        }
        
        Topology.Save(ostream);
        Input.Save(ostream);
        Output.Save(ostream);

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::complex::Lesson2D::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Lesson2D<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::complex::Lesson2D::Load(), istream.good() == false.");
        }

        decltype(Topology) topology;
        decltype(Input) input;
        decltype(Output) output;

        topology.Load(istream);
        CheckTopology(topology);

        if ((topology.GetInputSize().GetArea() != 0) && (topology.GetOutputCount() != 0))
        {
          input.Load(istream);
          if (input.GetSize() != topology.GetInputSize())
          {
            throw std::logic_error("cnn::engine::complex::Lesson2D::Load(), input.GetSize() != topology.GetInputSize().");
          }
          output.Load(istream);
          if (output.GetValueCount() != topology.GetOutputCount())
          {
            throw std::logic_error("cnn::engine::complex::Lesson2D::Load(), output.GetValueCount() != topology.GetOutputCount().");
          }
        }

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::complex::Lesson2D::Load(), istream.good() == false.");
        }

        Topology = std::move(topology);
        Input = std::move(input);
        Output = std::move(output);
      }

      template <typename T>
      void Lesson2D<T>::CheckTopology(const Lesson2DTopology& topology) const
      {
        // Zero topology is allowed.
        if ((topology.GetInputSize().GetArea() == 0) && (topology.GetOutputCount() == 0))
        {
          return;
        }

        if ((topology.GetInputSize().GetArea() == 0) || (topology.GetOutputCount() == 0))
        {
          throw std::invalid_argument("cnn::engine::complex::Lesson2D::CheckTopology(), (topology.GetInputSize().GetArea() == 0) || (topology.GetOutputCount() == 0).");
        }
      }
    }
  }
}