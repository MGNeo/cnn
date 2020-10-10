#pragma once

#include <filesystem>

#include "i_loader.hpp"
#include "3rd/SFML-2.5.1/include/SFML/Graphics.hpp"
#include "engine/complex/lesson_2d_library.hpp"

using namespace cnn::engine::complex;

namespace cnn
{
  namespace examples
  {
    namespace complex_learning
    {
      template <typename T>
      class Loader : public ILoader<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Loader();

        typename ILesson2DLibrary<T>::Uptr Load() const override;

      private:

        size_t InputWidth;
        size_t InputHeight;
        size_t InputCount;
        size_t OutputCount;

        void LoadNumbers(ILesson2DLibrary<T>& library) const;
        void LoadNumber(ILesson2DLibrary<T>& library, const size_t number) const;

      };

      template <typename T>
      Loader<T>::Loader()
        :
        InputWidth{ 32 },
        InputHeight{ 32 },
        InputCount{ 1 },
        OutputCount{ 10 }
      {
      }

      // Returned value can't have nullptr.
      template <typename T>
      typename engine::complex::ILesson2DLibrary<T>::Uptr Loader<T>::Load() const
      {
        
        auto library = std::make_unique<Lesson2DLibrary<T>>(InputWidth,
                                                            InputHeight,
                                                            InputCount,
                                                            OutputCount);

        LoadNumbers(*library);

        return std::move(library);
      }

      template <typename T>
      void Loader<T>::LoadNumbers(ILesson2DLibrary<T>& library) const
      {
        for (size_t number = 0; number < OutputCount; ++number)
        {
          LoadNumber(library, number);
        }
      }

      template <typename T>
      void Loader<T>::LoadNumber(ILesson2DLibrary<T>& library, const size_t number) const
      {
        if (number >= OutputCount)
        {
          throw std::invalid_argument("cnn::examplex::Loader::LoadNumber(), number >= OutputCount.");
        }

        const std::string path = "../../data/numbers/" + std::to_string(number) + "/";

        std::filesystem::directory_iterator di{ path };
        
        for (auto de : di)
        {
          if (de.is_regular_file() && (de.path().extension().string() == ".bmp"))
          {
            sf::Image image;
            if (image.loadFromFile(de.path().string()))
            {
              if ((image.getSize().x == InputWidth) && (image.getSize().y == InputHeight))
              {
                library.PushBack();
                auto& lesson = library.GetLastLesson();
                for (size_t x = 0; x < InputWidth; ++x)
                {
                  for (size_t y = 0; y < InputHeight; ++y)
                  {
                    T value{};

                    if (image.getPixel(static_cast<unsigned int>(x), static_cast<unsigned int>(y)) == sf::Color::Black)
                    {
                      value = 1;
                    }

                    lesson.GetInput(0).SetValue(x, y, value);
                  }
                }
                lesson.GetOutput().SetValue(number, 1);
              }
            }
          }
        }
      }

    }
  }
}

