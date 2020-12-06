#pragma once

#include "Network2DTopology.hpp"

#include "../convolution/Network2D.hpp"
#include "../perceptron/Network.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class Network2D
      {
      public:

        Network2D(const Network2DTopology& topology = {});

        Network2D(const Network2D& network) = default;

        Network2D(Network2D&& network) noexcept = default;

        Network2D& operator=(const Network2D& network);

        Network2D& operator=(Network2D&& network) noexcept = default;

        const Network2DTopology& GetTopology() const;

        void SetTopology(const Network2DTopology& topology);

        const convolution::Network2D<T>& GetConvolutionNetwork() const;

        convolution::Network2D<T>& GetConvolutionNetwork();

        const perceptron::Network<T>& GetPerceptronNetwork() const;

        perceptron::Network<T>& GetPerceptronNetwork();

        // ...

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

        Network2DTopology Topology;
        convolution::Network2D<T> ConvolutionNetwork;
        perceptron::Network<T> PerceptronNetwork;

        void CheckTopology(const Network2DTopology& topology) const;

      };

      template <typename T>
      Network2D<T>::Network2D(const Network2DTopology& topology)
      {
        CheckTopology(topology);

        Topology = topology;

        ConvolutionNetwork.SetTopology(Topology.GetConvolutionTopology());
        PerceptronNetwork.SetTopology(Topology.GetPerceptronTopology());
      }

      template <typename T>
      Network2D<T>& Network2D<T>::operator=(const Network2D& network)
      {
        if (this != &network)
        {
          Network2D<T> tmpNetwork{ network };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, tmpNetwork);
        }
        return *this;
      }

      template <typename T>
      const Network2DTopology& Network2D<T>::GetTopology() const
      {
        return Topology;
      }

      template <typename T>
      void Network2D<T>::SetTopology(const Network2DTopology& topology)
      {
        Network2D<T> tmpNetwork{ topology };
        // Beware, it is very intimate place for strong exception guarantee.
        std::swap(*this, tmpNetwork);
      }

      template <typename T>
      const convolution::Network2D<T>& Network2D<T>::GetConvolutionNetwork() const
      {
        return ConvolutionNetwork;
      }

      template <typename T>
      convolution::Network2D<T>& Network2D<T>::GetConvolutionNetwork()
      {
        return ConvolutionNetwork;
      }

      template <typename T>
      const perceptron::Network<T>& Network2D<T>::GetPerceptronNetwork() const
      {
        return PerceptronNetwork;
      }

      template <typename T>
      perceptron::Network<T>& Network2D<T>::GetPerceptronNetwork()
      {
        return PerceptronNetwork;
      }

      // ...

      template <typename T>
      void Network2D<T>::GenerateOutput()
      {
        ConvolutionNetwork.GenerateOputput();
        const auto& lastLayer = ConvolutionNetwork.GetLastLayer();

        //for (size_t o = 0; o < C)
      }

      template <typename T>
      void Network2D<T>::Clear() noexcept
      {
        ConvolutionNetwork.Clear();
        PerceptronNetwork.Clear();
      }

      template <typename T>
      void Network2D<T>::Reset() noexcept
      {
        Topology.Reset();
        ConvolutionNetwork.Reset();
        PerceptronNetwork.Reset();
      }

      template <typename T>
      void Network2D<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::complex::Network2D::Save(), ostream.good() == false.");
        }

        Topology.Save(ostream);
        ConvolutionNetwork.Save(ostream);
        PerceptronNetwork.Save(ostream);

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::complex::Network2D::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Network2D<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::complex::Network2D::Load(), istream.good() == false.");
        }

        decltype(Topology) topology;
        decltype(ConvolutionNetwork) convolutionNetwork;
        decltype(PerceptronNetwork) perceptronNetwork;

        topology.Load(istream);
        CheckTopology(topology);

        convolutionNetwork.Load(istream);
        perceptronNetwork.Load(istream);

        if (topology.GetConvolutionTopology() != convolutionNetwork.GetTopology())
        {
          throw std::logic_error("cnn::engine::complex::Network2D::Load(), topology.GetConvolutionTopology() != convolutionNetwork.GetTopology().");
        }
        if (topology.GetPerceptronTopology() != perceptronNetwork.GetTopology())
        {
          throw std::logic_error("cnn::engine::complex::Network2D::Load(), topology.GetPerceptronTopology() != perceptronNetwork.GetTopology().");
        }

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::complex::Network2D::Load(), istream.good() == false.");
        }
        
        Topology = std::move(topology);
        ConvolutionNetwork = std::move(convolutionNetwork);
        PerceptronNetwork = std::move(perceptronNetwork);
      }

      template <typename T>
      void Network2D<T>::FillWeights(common::ValueGenerator<T>& valueGenerator) noexcept
      {
        ConvolutionNetwork.FillWeights(valueGenerator);
        PerceptronNetwork.FillWeights(valueGenerator);
      }

      template <typename T>
      void Network2D<T>::Mutate(common::Mutagen<T>& mutagen) noexcept
      {
        ConvolutionNetwork.Mutate(mutagen);
        PerceptronNetwork.Mutate(mutagen);
      }

      template <typename T>
      void Network2D<T>::CheckTopology(const Network2DTopology& topology) const
      {
        // Zero topology is allowed.
        if ((topology.GetConvolutionTopology().GetLayerCount() == 0) && (topology.GetPerceptronTopology().GetLayerCount() == 0))
        {
          return;
        }

        if ((topology.GetConvolutionTopology().GetLayerCount() != 0) || (topology.GetPerceptronTopology().GetLayerCount() != 0))
        {
          throw std::invalid_argument("cnn::engine::complex::Network2D::CheckTopology(), (topology.GetConvolutionTopology().GetLayerCount() != 0) || (topology.GetPerceptronTopology().GetLayerCount() != 0).");
        }

        if (topology.GetConvolutionTopology().GetLastLayerTopology().GetOutputValueCount() != topology.GetPerceptronTopology().GetFirstLayerTopology().GetInputCount())
        {
          throw std::invalid_argument("cnn::engine::complex::Network2D::CheckTopology(), topology.GetConvolutionTopology().GetLastLayerTopology().GetOutputValueCount() != topology.GetPerceptronTopology().GetFirstLayerTopology().GetInputCount().");
        }
      }

    }
  }
}