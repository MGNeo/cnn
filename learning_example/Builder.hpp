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

      constexpr static size_t InputWidth = 32;
      constexpr static size_t InputHeight = 32;
      constexpr static size_t InputCount = 1;
      constexpr static size_t OutputCount = 10;

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
      // Convolution network topology.
      engine::convolution::Network2DTopology convolutionNetworkTopology;
      {
        // First layer topology.
        {
          engine::convolution::Layer2DTopology layerTopology;
          layerTopology.SetInputSize({ InputWidth, InputHeight });
          layerTopology.SetInputCount(1);
          layerTopology.SetFilterTopology({ { 3, 3 }, 1 });
          layerTopology.SetFilterCount(5);
          layerTopology.SetOutputSize({ 30, 30 });
          layerTopology.SetOutputCount(5);
          convolutionNetworkTopology.PushBack(layerTopology);
        }
        // Second layer topology.
        {
          engine::convolution::Layer2DTopology layerTopology;
          layerTopology.SetInputSize({ 30, 30 });
          layerTopology.SetInputCount(5);
          layerTopology.SetFilterTopology({ { 5, 5 }, 5 });
          layerTopology.SetFilterCount(10);
          layerTopology.SetOutputSize({ 26, 26 });
          layerTopology.SetOutputCount(10);
          convolutionNetworkTopology.PushBack(layerTopology);
        }
        // Third layer topology.
        {
          engine::convolution::Layer2DTopology layerTopology;
          layerTopology.SetInputSize({ 26, 26 });
          layerTopology.SetInputCount(10);
          layerTopology.SetFilterTopology({ { 7, 7 }, 10 });
          layerTopology.SetFilterCount(20);
          layerTopology.SetOutputSize({ 20, 20 });
          layerTopology.SetOutputCount(20);
          convolutionNetworkTopology.PushBack(layerTopology);
        }
        // Fourth layer
        {
          engine::convolution::Layer2DTopology layerTopology;
          layerTopology.SetInputSize({ 20, 20 });
          layerTopology.SetInputCount(20);
          layerTopology.SetFilterTopology({ { 5, 5 }, 20 });
          layerTopology.SetFilterCount(10);
          layerTopology.SetOutputSize({ 16, 16 });
          layerTopology.SetOutputCount(10);
          convolutionNetworkTopology.PushBack(layerTopology);
        }
      }

      // Perceptron network topology.
      engine::perceptron::NetworkTopology perceptronNetworkTopology;
      {
        // First layer topology.
        {
          engine::perceptron::LayerTopology layerTopology;
          layerTopology.SetInputCount(convolutionNetworkTopology.GetLastLayerTopology().GetOutputValueCount());
          layerTopology.SetNeuronCount(20);
          perceptronNetworkTopology.PushBack(layerTopology);
        }
        // Second layer topology.
        {
          engine::perceptron::LayerTopology layerTopology;
          layerTopology.SetInputCount(20);
          layerTopology.SetNeuronCount(15);
          perceptronNetworkTopology.PushBack(layerTopology);
        }
        // Third layer topology.
        {
          engine::perceptron::LayerTopology layerTopology;
          layerTopology.SetInputCount(15);
          layerTopology.SetNeuronCount(OutputCount);
          perceptronNetworkTopology.PushBack(layerTopology);
        }
      }

      // Complex network.
      engine::complex::Network2D<T> complexNetwork{ { convolutionNetworkTopology, perceptronNetworkTopology } };

      return complexNetwork;
    }

    template <typename T>
    typename engine::complex::GeneticAlgorithm2D<T> Builder<T>::GetGeneticAlgorithm()
    {
      return {};
    }

    template <typename T>
    engine::complex::Lesson2DLibrary<T> Builder<T>::LoadLessons()
    {
      engine::complex::Lesson2DLibrary<T> lessonLibrary;
      for (size_t number = 0; number < OutputCount; ++number)
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

      if (image.getSize().x != InputWidth)
      {
        throw std::runtime_error("cnn::learning_example::Builder::LoadLesson(), image.getSize().x != InputWidth.");
      }

      if (image.getSize().y != InputHeight)
      {
        throw std::runtime_error("cnn::learning_example::Builder::LoadLesson(), image.getSize().y != InputHeight.");
      }

      engine::complex::Lesson2D<T> lesson{ {{InputWidth, InputHeight}, InputCount, OutputCount} };
      auto input = lesson.GetInput(0);
      auto output = lesson.GetOutput();
      for (unsigned int x = 0; x < InputWidth; ++x)
      {
        for (unsigned int y = 0; y < InputHeight; ++y)
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
