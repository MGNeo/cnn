#pragma once

#include <type_traits>
#include <filesystem>
#include <SFML/Graphics.hpp>

#include "../engine/complex/Lesson2DLibrary.hpp"
#include "../engine/complex/Network2D.hpp"
#include "../engine/complex/GeneticAlgorithm2D.hpp"

namespace cnn
{
  namespace learning_example
  {
    template <typename T>
    class Builder
    {

      static_assert(std::is_floating_point<T>::value);

    public:

      static typename engine::complex::Lesson2DLibrary<T> GetLessonLibrary();
      static typename engine::complex::Network2D<T> GetNetwork();
      static typename engine::complex::GeneticAlgorithm2D<T> GetGeneticAlgorithm();

    private:

      ~Builder() = delete;

      static engine::complex::Lesson2DLibrary<T> LoadLessons();
      static engine::complex::Lesson2D<T> LoadLesson(const std::string& fileName, const size_t number);

    };

    template <typename T>
    typename engine::complex::Lesson2DLibrary<T> Builder<T>::GetLessonLibrary()
    {
      return LoadLessons();
    }

    template <typename T>
    typename engine::complex::Network2D<T> Builder<T>::GetNetwork()
    {
      // ...
      return {};
    }

    template <typename T>
    typename engine::complex::GeneticAlgorithm2D<T> Builder<T>::GetGeneticAlgorithm()
    {
      // ...
      return {};
    }

    template <typename T>
    engine::complex::Lesson2DLibrary<T> Builder<T>::LoadLessons()
    {
      engine::complex::Lesson2DLibrary<T> lessonLibrary;
      for (size_t number = 0; number <= 9; ++number)
      {
        const std::string path = "../data/numbers/" + std::to_string(number);
        std::filesystem::directory_iterator di(path);

        for (auto& d : di)
        {
          if (d.is_directory() == false)
          {
            const std::string extension = d.path().extension().string();
            if (extension == ".bmp")
            {
              const std::string fileName = d.path().string();

              // Standard allows us to do this.
              const engine::complex::Lesson2D<T>& lesson = LoadLesson(fileName, number);

              lessonLibrary.PushBack(lesson);
            }
          }
        }
      }
      return lessonLibrary;
    }

    template <typename T>
    engine::complex::Lesson2D<T> Builder<T>::LoadLesson(const std::string& fileName, const size_t number)
    {
      sf::Image image;
      const bool imageWasLoaded = image.loadFromFile(fileName);

      if (imageWasLoaded == false)
      {
        throw std::runtime_error("cnn::learning_example::Builder::LoadLesson(), imageWasLoaded == false.");
      }

      const size_t expectedWidth = 32;
      const size_t expectedHeight = 32;

      if (image.getSize().x != expectedWidth)
      {
        throw std::runtime_error("cnn::learning_example::Builder::LoadLesson(), image.getSize().x != expectedWidth.");
      }

      if (image.getSize().y != expectedHeight)
      {
        throw std::runtime_error("cnn::learning_example::Builder::LoadLesson(), image.getSize().y != expectedHeight.");
      }

      engine::complex::Lesson2D<T> lesson{ {{32, 32}, 1, 10} };
      auto input = lesson.GetInput(0);
      auto output = lesson.GetOutput();
      for (unsigned int x = 0; x < expectedWidth; ++x)
      {
        for (unsigned int y = 0; y < expectedHeight; ++y)
        {
          T value{};
          if ((image.getPixel(x, y).r == 0) && (image.getPixel(x, y).g == 0) && (image.getPixel(x, y).b == 0))
          {
            value = 1;
          }
          input.SetValue(x, y, value);
        }
      }
      output.SetValue(number, 1);

      return lesson;
    }
  }
}
