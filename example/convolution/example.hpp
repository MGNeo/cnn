#pragma once

#include "../common/i_example.hpp"
#include "../../engine/convolution/network_2d.hpp"
#include "layer_2d_visitor.hpp"

#include <random>
#include <time.h>

namespace cnn
{
  namespace example
  {
    namespace convolution
    {
      template <typename T>
      class Example : public common::IExample<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<Example<T>>;

        void Execute() const override;

      private:

        void Simple() const;
        void Visitor() const;

      };

      template <typename T>
      void Example<T>::Execute() const
      {
        std::cout << __FUNCSIG__ << std::endl;
        {
          Simple();
          Visitor();
        }
        std::cout << std::endl;

        /*
        // Visit all layers of the network.
        {
          typename engine::convolution::ILayer2DVisitor<T>::Uptr layerVisitor2D = std::make_unique<Layer2DVisitor<T>>();
          for (size_t l = 0; l < network2D->GetLayerCount(); ++l)
          {
            network2D->GetLayer(l).Accept(*layerVisitor2D);
          }
        }
        */
      }

      template <typename T>
      void Example<T>::Simple() const
      {
        std::cout << "  " << __FUNCSIG__ << std::endl;
        {
          // Create new 2D convolution network.
          auto network2D = std::make_unique<engine::convolution::Network2D<T>>(32, 32, 3, 4, 4, 5);

          // Add few layers to the network.
          {
            // Pooling layer.
            network2D->PushBack(3);
            // Convolution layer.
            network2D->PushBack(5, 5, 15);
          }

          // Set random signals in first layer of the network.
          {
            std::default_random_engine dre{ static_cast<unsigned int>(time(NULL)) };
            std::uniform_real_distribution<T> urd{ -1.f, +1.f };

            auto& firstLayer = network2D->GetLayer(0);

            for (size_t i = 0; i < firstLayer.GetInputCount(); ++i)
            {
              for (size_t x = 0; x < firstLayer.GetInputWidth(); ++x)
              {
                for (size_t y = 0; y < firstLayer.GetInputHeight(); ++y)
                {
                  firstLayer.GetInput(i).SetValue(x, y, urd(dre));
                }
              }
            }
          }

          // Move input signal through the network.
          {
            network2D->Process();
          }

          // Take the result from the last layer.
          {
            const auto& lastLayer = network2D->GetLastLayer();
            // ...
          }
        }
        std::cout << std::endl;
      }

      template <typename T>
      void Example<T>::Visitor() const
      {
        std::cout << "  " << __FUNCSIG__ << std::endl;
        {
          auto network2D = std::make_unique<engine::convolution::Network2D<T>>(32, 32, 3, 4, 4, 5);

          // Pooling.
          network2D->PushBack(2);
          // Convolution.
          network2D->PushBack(3, 3, 25);
          // Pooling.
          network2D->PushBack(3);
          // Convolution.
          network2D->PushBack(4, 4, 15);

          auto layer2DVisitor = std::make_unique<Layer2DVisitor<T>>();

          network2D->Accept(*layer2DVisitor);

        }
        std::cout << std::endl;
      }
    }
  }
}