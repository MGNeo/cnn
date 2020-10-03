#pragma once

#include "i_lesson_2d.hpp"

#include "../convolution/map_2d.hpp"
#include "../common/i_map.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class Lesson2D : public ILesson2D<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<Lesson2D<T>>;

        Lesson2D(const size_t inputWidth,
                 const size_t inputHeight,
                 const size_t inputCount,
                 const size_t outputSize);

        size_t GetInputWidth() const override;
        size_t GetInputHeight() const override;
        size_t GetInputCount() const override;

        const convolution::IMap2D<T>& GetInput(const size_t index) const override;
        convolution::IMap2D<T>& GetInput(const size_t index) override;

        size_t GetOutputSize() const override;

        const common::IMap<T>& GetOutput() const override;
        common::IMap<T>& GetOutput() override;

      private:

        size_t InputWidth;
        size_t InputHeight;
        size_t InputCount;
        std::unique_ptr<typename convolution::IMap2D<T>::Uptr[]> Inputs;

        typename common::IMap<T>::Uptr Output;

      };

      template <typename T>
      Lesson2D<T>::Lesson2D(const size_t inputWidth,
                            const size_t inputHeight,
                            const size_t inputCount,
                            const size_t outputSize)
        :
        InputWidth{ inputWidth },
        InputHeight{ inputHeight },
        InputCount{ inputCount }
      {
        if (InputCount == 0)
        {
          throw std::invalid_argument("cnn::engine::complex::Lesson2D<T>::Lesson2D(), InputCount == 0.");
        }
        Inputs = std::make_unique<typename convolution::Map2D<T>::Uptr[]>(InputCount);
        for (size_t i = 0; i < InputCount; ++i)
        {
          Inputs[i] = std::make_unique<convolution::Map2D<T>>(InputWidth,
                                                              InputHeight);
        }
        Output = std::make_unique<common::Map<T>>(outputSize);
      }

      template <typename T>
      size_t Lesson2D<T>::GetInputWidth() const
      {
        return InputWidth;
      }

      template <typename T>
      size_t Lesson2D<T>::GetInputHeight() const
      {
        return InputHeight;
      }

      template <typename T>
      size_t Lesson2D<T>::GetInputCount() const
      {
        return InputCount;
      }

      template <typename T>
      const convolution::IMap2D<T>& Lesson2D<T>::GetInput(const size_t index) const
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::complex::Lesson2D::GetInput() const, index >= InputCount.");
        }
        return *(Inputs[index]);
      }

      template <typename T>
      convolution::IMap2D<T>& Lesson2D<T>::GetInput(const size_t index)
      {
        if (index >= InputCount)
        {
          throw std::range_error("cnn::engine::complex::Lesson2D::GetInput(), index >= InputCount.");
        }
        return *(Inputs[index]);
      }

      template <typename T>
      size_t Lesson2D<T>::GetOutputSize() const
      {
        return Output->GetValueCount();
      }

      template <typename T>
      const common::IMap<T>& Lesson2D<T>::GetOutput() const
      {
        return *Output;
      }

      template <typename T>
      common::IMap<T>& Lesson2D<T>::GetOutput()
      {
        return *Output;
      }
    }
  }
}
