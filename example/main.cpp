#include <iostream>
#include <random>

#include "perceptron/example.hpp"
#include "convolution/example.hpp"
#include "complex/example.hpp"

// 1 TODO: Create a genetic algorithm.
// 2 TODO: Add methods which will be exception safety (such methods must use "copy-swap" idiom).
// 3 TODO: Extend examples for sub examples.
// 4 TODO: Create special builders for creating of networks, etc.

int main()
{
  try
  {
    // Examples of using for cnn::engine::perceptron.
    {
      {
        cnn::example::common::IExample<float>::Uptr example = std::make_unique<cnn::example::perceptron::Example<float>>();
        example->Execute();
      }

      {
        cnn::example::common::IExample<double>::Uptr example = std::make_unique<cnn::example::perceptron::Example<double>>();
        example->Execute();
      }

      {
        cnn::example::common::IExample<long double>::Uptr example = std::make_unique<cnn::example::perceptron::Example<long double>>();
        example->Execute();
      }
    }

    // Examples of using for cnn::engine::convolution.
    {
      {
        cnn::example::common::IExample<float>::Uptr example = std::make_unique<cnn::example::convolution::Example<float>>();
        example->Execute();
      }

      {
        cnn::example::common::IExample<double>::Uptr example = std::make_unique<cnn::example::convolution::Example<double>>();
        example->Execute();
      }

      {
        cnn::example::common::IExample<long double>::Uptr example = std::make_unique<cnn::example::convolution::Example<long double>>();
        example->Execute();
      }
    }

    // Examples of using for cnn::engine::complex.
    {
      {
        cnn::example::common::IExample<float>::Uptr example = std::make_unique<typename cnn::example::complex::Example<float>>();
        example->Execute();
      }

      {
        cnn::example::common::IExample<double>::Uptr example = std::make_unique<cnn::example::complex::Example<double>>();
        example->Execute();
      }

      {
        cnn::example::common::IExample<long double>::Uptr example = std::make_unique<cnn::example::complex::Example<long double>>();
        example->Execute();
      }
    }
  }
  catch (const std::exception& exception)
  {
    std::cout << "cnn::example::main(), std::exception was caught: " << exception.what() << std::endl;
    return EXIT_FAILURE;
  }
  catch (...)
  {
    std::cout << "cnn::example::main(), unknown exception was caught." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "All was successfully completed!" << std::endl;
  return EXIT_SUCCESS;
}

