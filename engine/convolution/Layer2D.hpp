#pragma once

#include "Layer2DTopology.hpp"

#include "Map2D.hpp"
//#include "ProxyMap2D.hpp"

#include "Filter2D.hpp"
//#include "ProxyFilter2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace convolution
    {
      template <typename T>
      class Layer2D
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Layer2D(const Layer2DTopology& topology = {});

        Layer2D(const Layer2D& layer);

        Layer2D(Layer2D&& layer) noexcept = default;

        Layer2D& operator=(const Layer2D& layer);

        Layer2D& operator=(Layer2D&& layer) noexcept = default;

        Layer2DTopology GetTopology() const noexcept;

        void SetTopology(const Layer2DTopology& topology);

        //ProxyMap2D<T> GetInput(const size_t index);
        
        //ProxyFilter2D<T> GetFilter(const size_t index);

        //ProxyMap2D<T> GetOutput(const size_t index);

        // Exception guarantee: base for this.
        void GenerateOutput();

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

        Layer2DTopology Topology;

        std::unique_ptr<Map2D<T>[]> Inputs;
        std::unique_ptr<Filter2D<T>[]> Filters;
        std::unique_ptr<Map2D<T>[]> Outputs;

        void CheckTopology(const Layer2DTopology& topology) const;

      };

      template <typename T>
      Layer2D<T>::Layer2D(const Layer2DTopology& topology)
      {
        CheckTopology(topology);

        Topology = topology;

        if (Topology.GetInputCount() != 0)
        {
          Inputs = std::make_unique<Map2D<T>[]>(Topology.GetInputCount());
          for (size_t i = 0; i < Topology.GetInputCount(); ++i)
          {
            Inputs[i].SetSize(Topology.GetInputSize());
          }
        }

        if (Topology.GetFilterCount() != 0)
        {
          Filters = std::make_unique<Filter2D<T>[]>(Topology.GetFilterCount());
          for (size_t i = 0; i < Topology.GetFilterCount(); ++i)
          {
            Filters[i].SetTopology(Topology.GetFilterTopology());
          }
        }

        if (Topology.GetOutputCount() != 0)
        {
          Outputs = std::make_unique<Map2D<T>[]>(Topology.GetOutputCount());
          for (size_t i = 0; i < Topology.GetOutputCount(); ++i)
          {
            Outputs[i].SetSize(Topology.GetOutputSize());
          }
        }
      }

      template <typename T>
      Layer2D<T>::Layer2D(const Layer2D& layer)
        :
        Topology{ layer.Topology }
      {
        if (Topology.GetInputCount() != 0)
        {
          Inputs = std::make_unique<Map2D<T>[]>(Topology.GetInputCount());
          for (size_t i = 0; i < Topology.GetInputCount(); ++i)
          {
            Inputs[i] = layer.Inputs[i];
          }
        }

        if (Topology.GetFilterCount() != 0)
        {
          Filters = std::make_unique<Filter2D<T>[]>(Topology.GetFilterCount());
          for (size_t i = 0; i < Topology.GetFilterCount(); ++i)
          {
            Filters[i] = layer.Filters[i];
          }
        }

        if (Topology.GetOutputCount() != 0)
        {
          Outputs = std::make_unique<Map2D<T>[]>(Topology.GetOutputCount());
          for (size_t i = 0; i < Topology.GetOutputCount(); ++i)
          {
            Outputs[i] = layer.Outputs[i];
          }
        }
      }

      template <typename T>
      Layer2D<T>& Layer2D<T>::operator=(const Layer2D<T>& layer)
      {
        if (this != &layer)
        {
          Layer2D<T> tmpLayer{ layer };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpLayer);
        }
        return *this;
      }

      template <typename T>
      Layer2DTopology Layer2D<T>::GetTopology() const noexcept
      {
        return Topology;
      }

      template <typename T>
      void Layer2D<T>::SetTopology(const Layer2DTopology& topology)
      {
        Layer2D<T> tmpLayer{ topology };
        // Beware, it is very intimate place for strong exception guarantee.
        std::swap(*this, tmpLayer);
      }

      /*
      template <typename T>
      ProxyMap2D<T> Layer2D<T>::GetInput(const size_t index)
      {
        if (index >= Topology.GetInputCount())
        {
          throw std::range_error("cnn::engine::convolution::Layer2D::GetInput(), index >= Topology.GetInputCount().");
        }
        return Inputs[index];
      }

      template <typename T>
      ProxyFilter2D<T> Layer2D<T>::GetFilter(const size_t index)
      {
        if (index >= Topology.GetFilterCount())
        {
          throw std::range_error("cnn::engine::convolution::Layer2D::GetFilter(), index >= Topology.GetFilterCount().");
        }
        return Filters[index];
      }

      template <typename T>
      ProxyMap2D<T> Layer2D<T>::GetOutput(const size_t index)
      {
        if (index >= Topology.GetOutputCount())
        {
          throw std::range_error("cnn::engine::convolution::Layer2D::GetOutput(), index >= Topology.GetOutputCount().");
        }
        return Outputs[index];
      }
      */

      template <typename T>
      void Layer2D<T>::GenerateOutput()
      {
        /*
        for (size_t f = 0; f < Topology.GetFilterCount(); ++f)
        {
          auto& output = Outputs[f];
          output.Clear();

          auto& filter = Filters[f];
          for (size_t c = 0; c < Topology.GetFilterTopology().GetCoreCount(); ++c)
          {
            const auto& input = Inputs[c];
            auto core = filter.GetCore(c);

            for (size_t ox = 0; ox < Topology.GetOutputSize().GetWidth(); ++ox)
            {
              for (size_t oy = 0; oy < Topology.GetOutputSize().GetHeight(); ++oy)
              {
                for (size_t cx = 0; cx < Topology.GetFilterTopology().GetSize().GetWidth(); ++cx)
                {
                  for (size_t cy = 0; cy < Topology.GetFilterTopology().GetSize().GetHeight(); ++cy)
                  {
                    const T value = input.GetValue(ox + cx, oy + cy);
                    core.SetInput(cx, cy, value);
                  }
                }
                core.GenerateOutput();
                output.SetValue(ox, oy, core.GetOutput());
              }
            }
          }
        }
        */
      }

      template <typename T>
      void Layer2D<T>::Clear() noexcept
      {
        for (size_t i = 0; i < Topology.GetInputCount(); ++i)
        {
          Inputs[i].Clear();
        }

        for (size_t i = 0; i < Topology.GetFilterCount(); ++i)
        {
          Filters[i].Clear();
        }

        for (size_t i = 0; i < Topology.GetOutputCount(); ++i)
        {
          Outputs[i].Clear();
        }
      }

      template <typename T>
      void Layer2D<T>::Reset() noexcept
      {
        Topology.Clear();
        Inputs.reset(nullptr);
        Filters.reset(nullptr);
        Outputs.reset(nullptr);
      }

      template <typename T>
      void Layer2D<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Save(), ostream.good() == false.");
        }

        Topology.Save(ostream);

        for (size_t i = 0; i < Topology.GetInputCount(); ++i)
        {
          Inputs[i].Save(ostream);
        }

        for (size_t i = 0; i < Topology.GetFilterCount(); ++i)
        {
          Filters[i].Save(ostream);
        }

        for (size_t i = 0; i < Topology.GetOutputCount(); ++i)
        {
          Outputs[i].Save(ostream);
        }

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Layer2D::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Layer2D<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::Load(), istream.good() == false.");
        }

        decltype(Topology) topology;
        decltype(Inputs) inputs;
        decltype(Filters) filters;
        decltype(Outputs) outputs;

        topology.Load(istream);

        CheckTopology(topology);

        if (topology.GetInputCount() != 0)
        {
          inputs = std::make_unique<Map2D<T>[]>(topology.GetInputCount());
          for (size_t i = 0; i < topology.GetInputCount(); ++i)
          {
            inputs[i].Load(istream);
            if (inputs[i].GetSize() != topology.GetInputSize())
            {
              throw std::logic_error("cnn::engine::convolution::Layer2D::Load(), inputs[i].GetSize() != topology.GetInputSize().");
            }
          }
        }

        if (topology.GetFilterCount() != 0)
        {
          filters = std::make_unique<Filter2D<T>[]>(topology.GetFilterCount());
          for (size_t i = 0; i < topology.GetFilterCount(); ++i)
          {
            filters[i].Load(istream);
            if (filters[i].GetTopology() != topology.GetFilterTopology())
            {
              throw std::logic_error("cnn::engine::convolution::Layer2D::Load(), filters[i].GetTopology() != topology.GetFilterTopology().");
            }
          }
        }

        if (topology.GetOutputCount() != 0)
        {
          outputs = std::make_unique<Map2D<T>[]>(topology.GetOutputCount());
          for (size_t i = 0; i < topology.GetOutputCount(); ++i)
          {
            outputs[i].Load(istream);
            if (outputs[i].GetSize() != topology.GetOutputSize())
            {
              throw std::logic_error("cnn::engine::convolution::Layer2D::Load(), outputs[i].GetSize() != topology.GetOutputSize().");
            }
          }
        }

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::convolution::Layer2D::Load(), istream.good() == false.");
        }

        Topology = std::move(topology);
        Inputs = std::move(inputs);
        Filters = std::move(filters);
        Outputs = std::move(outputs);
      }

      template <typename T>
      void Layer2D<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept
      {
        for (size_t i = 0; i < Topology.GetFilterCount(); ++i)
        {
          Filters[i].FillWeights(valueGenerator);
        }
      }

      template <typename T>
      void Layer2D<T>::Mutate(common::Mutagen<T>& mutagen) noexcept
      {
        for (size_t i = 0; i < Topology.GetFilterCount(); ++i)
        {
          Filters[i].Mutate(mutagen);
        }
      }

      template <typename T>
      void Layer2D<T>::CheckTopology(const Layer2DTopology& topology) const
      {
        // Zero topology is allowed.
        if ((topology.GetInputCount() == 0) && (topology.GetFilterCount() == 0) && (topology.GetOutputCount() == 0))
        {
          return;
        }

        if ((topology.GetInputCount() == 0) || (topology.GetFilterCount() == 0) || (topology.GetOutputCount() == 0))
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::CheckTopology(), wrong topology.");
        }

        if (topology.GetInputCount() != topology.GetFilterTopology().GetCoreCount())
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::CheckTopology(), topology.GetInputCount() != topology.GetFilterTopology().GetCoreCount().");
        }

        if (topology.GetFilterCount() != topology.GetOutputCount())
        {
          throw std::invalid_argument("cnn::engine::convolution::Layer2D::CheckTopology(), topology.GetFilterCount() != topology.GetOutputCount().");
        }

        // Width's
        {
          if (topology.GetInputSize().GetWidth() <= 1)
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CheckTopology(), topology.GetInputSize().GetWidth() <= 1.");
          }

          if (topology.GetFilterTopology().GetSize().GetWidth() <= 1)
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CheckTopology(), topology.GetFilterTopology().GetSize().GetWidth() <= 1.");
          }

          if (topology.GetOutputSize().GetWidth() == 0)
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CheckTopology(), topology.GetOutputSize().GetWidth() == 0.");
          }

          if ((topology.GetInputSize().GetWidth() - topology.GetFilterTopology().GetSize().GetWidth() + 1) != topology.GetOutputSize().GetWidth())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CheckTopology(), (topology.GetInputSize().GetWidth() - topology.GetFilterTopology().GetSize().GetWidth() + 1) != topology.GetOutputSize().GetWidth().");
          }
        }

        // Height's
        {
          if (topology.GetInputSize().GetHeight() <= 1)
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CheckTopology(), topology.GetInputSize().GetHeight() <= 1.");
          }

          if (topology.GetFilterTopology().GetSize().GetHeight() <= 1)
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CheckTopology(), topology.GetFilterTopology().GetSize().GetHeight() <= 1.");
          }

          if (topology.GetOutputSize().GetHeight() == 0)
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CheckTopology(), topology.GetOutputSize().GetHeight() == 0.");
          }

          if ((topology.GetInputSize().GetHeight() - topology.GetFilterTopology().GetSize().GetHeight() + 1) != topology.GetOutputSize().GetHeight())
          {
            throw std::invalid_argument("cnn::engine::convolution::Layer2D::CheckTopology(), (topology.GetInputSize().GetHeight() - topology.GetFilterTopology().GetSize().GetHeight() + 1) != topology.GetOutputSize().GetHeight().");
          }
        }
      }
    }
  }
}