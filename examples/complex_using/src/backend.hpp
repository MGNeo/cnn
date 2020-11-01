#pragma once

#include <string>
#include <engine/complex/network_2d.hpp>
#include <engine/convolution/network_2d.hpp>
#include <engine/perceptron/network.hpp>

#include "field.hpp"
#include "cursor.hpp"

namespace cnn
{
  namespace examples
  {
    namespace complex_using
    {
      template <typename T>
      class Backend
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Backend(const std::string& field,
                const std::string& network);

        size_t GetCursorX() const;
        size_t GetCursorY() const;

        void SetCursorX(const size_t x);
        void SetCursorY(const size_t y);

        void UpdateAnswers();

        size_t GetAnswerCount() const;
        T GetAnswer(const size_t index) const;

      private:

        void LoadField(const std::string& fileName);
        void LoadNetwork(const std::string& fileName);

        Field<T> Field_;
        Cursor Cursor_;
        typename engine::complex::INetwork2D<T>::Uptr Network;

      };

      template <typename T>
      Backend<T>::Backend(const std::string& field,
                          const std::string& network)
      {
        LoadField(field);
        LoadNetwork(network);
      }

      template <typename T>
      size_t Backend<T>::GetCursorX() const
      {
        return Cursor_.GetX();
      }

      template <typename T>
      size_t Backend<T>::GetCursorY() const
      {
        return Cursor_.GetY();
      }

      template <typename T>
      void Backend<T>::SetCursorX(const size_t x)
      {
        Cursor_.SetX(x);
      }

      template <typename T>
      void Backend<T>::SetCursorY(const size_t y)
      {
        Cursor_.SetY(y);
      }

      template <typename T>
      void Backend<T>::UpdateAnswers()
      {
        // ...
      }

      template <typename T>
      size_t Backend<T>::GetAnswerCount() const
      {
        // ...
      }

      template <typename T>
      T Backend<T>::GetAnswer(const size_t index) const
      {
        // ...
        return 0;
      }

      template <typename T>
      void Backend<T>::LoadField(const std::string& fileName)
      {
        Field_.LoadFromImage(fileName);
      }

      template <typename T>
      void Backend<T>::LoadNetwork(const std::string& fileName)
      {
        // Prepare convolution network.
        auto convolutionNetwork = std::make_unique<engine::convolution::Network2D<T>>(32, 32, 1, 3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(3, 3, 10);
        convolutionNetwork->PushBack(2, 2, 10);

        // Prepare perceptron network.
        auto perceptronNetwork = std::make_unique<engine::perceptron::Network<T>>(convolutionNetwork->GetLastLayer().GetOutputValueCount(), 15);
        perceptronNetwork->PushBack(10);

        // Prepare complex network.
        auto complexNetwork = std::make_unique<engine::complex::Network2D<T>>(std::move(convolutionNetwork), std::move(perceptronNetwork));

        // Load weights from the file.
        complexNetwork->Load(fileName);

        Network = std::move(complexNetwork);
      }
    }
  }
}