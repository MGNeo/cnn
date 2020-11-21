#pragma once

#include <memory>
#include <type_traits>
#include <stdexcept>

#include "Core2D.hpp"
#include "ProxyCore2D.hpp"
#include "Filter2DTopology.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Filter2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Filter2D(const Filter2DTopology& topology = {});

        Filter2D(const Filter2D& filter) = default;

        Filter2D(Filter2D&& filter) noexcept = default;

        // Exception guarantee: strong for this.
        Filter2D& operator=(const Filter2D& filter);

        Filter2D& operator=(Filter2D&& filter) noexcept = default;

        Filter2DTopology GetTopology() const noexcept;

        // Exception guarantee: strong for this.
        void SetTopology(const Filter2DTopology& topology);

        // Exception guarantee: strong for this.
        ProxyCore2D<T> GetCore(const size_t index) const;

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

        // We expect that the method never throws any exception.
        void FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept;

        // We expect that the method never throws any exception.
        void Mutate(common::Mutagen<T>& mutagen) noexcept;

      private:

        Filter2DTopology Topology;
        std::unique_ptr<Core2D<T>[]> Cores;

      };

      template <typename T>
      Filter2D<T>::Filter2D(const Filter2DTopology& topology)
        :
        Topology{ topology }
      {
        if (Topology.GetCoreCount() != 0)
        {
          Cores = std::make_unique<Core2D<T>[]>(Topology.GetCoreCount());
          if (Topology.GetSize().GetArea() != 0)
          {
            for (size_t i = 0; i < Topology.GetCoreCount(); ++i)
            {
              Cores[i].SetSize(Topology.GetSize());
            }
          }
        }
      }

      template <typename T>
      Filter2D<T>& Filter2D<T>::operator=(const Filter2D& filter)
      {
        if (this != &filter)
        {
          Filter2D<T> tmpFilter{ filter };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpFilter);
        }
        return *this;
      }

      template <typename T>
      Filter2DTopology Filter2D<T>::GetTopology() const noexcept
      {
        return Topology;
      }

      template <typename T>
      void Filter2D<T>::SetTopology(const Filter2DTopology& topology)
      {
        Filter2D<T> tmpFilter{ topology };
        // Beware, it is very intimate place for strong exception guarantee.
        std::swap(*this, tmpFilter);
      }

      template <typename T>
      ProxyCore2D<T> Filter2D<T>::GetCore(const size_t index) const
      {
        if (index >= Topology.GetCoreCount())
        {
          throw std::range_error("cnn::engine::convolution::Filter2D::SetTopology(), index >= Topology.GetCoreCount().");
        }
        return Cores[index];
      }

      template <typename T>
      void Filter2D<T>::Clear() noexcept
      {
        Topology.Clear();
      }

      template <typename T>
      void Filter2D<T>::Reset() noexcept
      {
        Topology.Clear();
        Cores.reset(nullptr);
      }

      template <typename T>
      void Filter2D<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Filter2D::Save(), ostream.good() == false.");
        }
        Topology.Save(ostream);
        for (size_t i = 0; i < Topology.GetCoreCount(); ++i)
        {
          Cores[i].Save(ostream);
        }
        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Filter2D::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Filter2D<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Filter2D::Load(), istream.good() == false.");
        }

        decltype(Topology) topology;
        decltype(Cores) cores;

        topology.Load(istream);
        
        if (topology.GetCoreCount() != 0)
        {
          // TODO: How can we change Core2D<T>[] on decltype()?
          cores = std::make_unique<Core2D<T>[]>(topology.GetCoreCount());
          for (size_t i = 0; i < topology.GetCoreCount(); ++i)
          {
            Cores[i].Load(istream);
            if (Cores[i].GetSize() != topology.GetSize())
            {
              throw std::logic_error("cnn::engine::convolution::Filter2D::Load(), Cores[i].GetSize() != topology.GetSize().");
            }
          }
        }

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Filter2D::Load(), istream.good() == false.");
        }

        Topology = std::move(topology);
        Cores = std::move(cores);

      }

      template <typename T>
      void Filter2D<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept
      {
        for (size_t i = 0; i < Topology.GetCoreCount(); ++i)
        {
          Cores[i].FillWeights(valueGenerator);
        }
      }

      template <typename T>
      void Filter2D<T>::Mutate(common::Mutagen<T>& mutagen) noexcept
      {
        for (size_t i = 0; i < Topology.GetCoreCount(); ++i)
        {
          Cores[i].Mutate(mutagen);
        }
      }
    }
  }
}
