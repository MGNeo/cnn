#pragma once

#include "../common/i_example.hpp"

#include "../../engine/perceptron/network.hpp"

#include "layer_visitor.hpp"

#include <random>
#include <time.h>

namespace cnn
{
  namespace example
  {
    namespace perceptron
    {
      template <typename T>
      class Example : public common::IExample<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<Example<T>>;
        
        void Execute() const override;
        
      };

      template <typename T>
      void Example<T>::Execute() const
      {
        // TODO: Write few private submethods for an each special example.
        // - Simple();
        // - Visitor();
        // - GeneticAlgorithm();
        // - etc.

        std::cout << __FUNCSIG__ << std::endl;

        // Create new perceptron network (one of implementations of cnn::engine::perceptron::INetwork).
        auto network = std::make_unique<engine::perceptron::Network<T>>(3, 8);

        // Add few layers to the network (topology of the network is ->3-[8]-[15]-[5]-[3]->).
        {
          network->PushBack(15);
          network->PushBack(5);
          network->PushBack(3);
        }

        // Set random signals in first layer of the network.
        {
          std::default_random_engine dre{ static_cast<unsigned int>(time(NULL)) };
          std::uniform_real_distribution<T> urd{ -1.f, +1.f };

          auto& firstLayer = network->GetLayer(0);

          for (size_t i = 0; i < firstLayer.GetInputSize(); ++i)
          {
            firstLayer.GetInput().SetValue(i, urd(dre));
          }
        }

        // Move input signal through the network.
        {
          network->Process();
        }

        // Visit all layers of the network.
        {
          typename engine::perceptron::ILayerVisitor<T>::Uptr layerVisitor = std::make_unique<LayerVisitor<T>>();
          for (size_t l = 0; l < network->GetLayerCount(); ++l)
          {
            network->GetLayer(l).Accept(*layerVisitor);
          }
        }

        std::cout << std::endl;
      }
    }
  }
}