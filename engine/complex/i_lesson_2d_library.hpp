#pragma once

#include <type_traits>
#include <memory>

#include "i_lesson_2d.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class ILesson2DLibrary
      {

        static_assert(std::is_floating_point<T>::value);

        public:

          using Uptr = std::unique_ptr<ILesson2DLibrary<T>>;

          virtual size_t GetLessonInputWidth() const = 0;
          virtual size_t GetLessonInputHeight() const = 0;
          virtual size_t GetLessonInputCount() const = 0;
          virtual size_t GetLessonOutputSize() const = 0;

          virtual size_t GetLessonCount() const = 0;

          virtual void PushBack() = 0;

          virtual const ILesson2D<T>& GetLesson(const size_t index) const = 0;
          virtual ILesson2D<T>& GetLesson(const size_t index) = 0;

          virtual const ILesson2D<T>& GetLastLesson() const = 0;
          virtual ILesson2D<T>& GetLastLesson() = 0;

          virtual ~ILesson2DLibrary() = default;

      };
    }
  }
}