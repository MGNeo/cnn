#pragma once

#include "i_lesson_2d_library.hpp"
#include "lesson_2d.hpp"

#include <vector>

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class Lesson2DLibrary : public ILesson2DLibrary<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        using Uptr = std::unique_ptr<Lesson2DLibrary<T>>;

        Lesson2DLibrary(const size_t lessonInputWidth,
                        const size_t lessonInputHeight,
                        const size_t lessonInputCount,
                        const size_t lessonOutputSize);

        size_t GetLessonInputWidth() const override;
        size_t GetLessonInputHeight() const override;
        size_t GetLessonInputCount() const override;
        size_t GetLessonOutputSize() const override;

        size_t GetLessonCount() const override;

        void PushBack() override;

        const ILesson2D<T>& GetLesson(const size_t index) const override;
        ILesson2D<T>& GetLesson(const size_t index) override;

        const ILesson2D<T>& GetLastLesson() const override;
        ILesson2D<T>& GetLastLesson() override;

      private:

        size_t LessonInputWidth;
        size_t LessonInputHeight;
        size_t LessonInputCount;
        size_t LessonOutputSize;

        std::vector<typename ILesson2D<T>::Uptr> Lessons;

      };

      template <typename T>
      Lesson2DLibrary<T>::Lesson2DLibrary(const size_t lessonInputWidth,
                                          const size_t lessonInputHeight,
                                          const size_t lessonInputCount,
                                          const size_t lessonOutputSize)
        :
        LessonInputWidth{ lessonInputWidth },
        LessonInputHeight{ lessonInputHeight },
        LessonInputCount{ lessonInputCount },
        LessonOutputSize{ lessonOutputSize }
      {
        if (LessonInputWidth == 0)
        {
          throw std::invalid_argument("cnn::engine::complex::Lesson2DLibrary::Lesson2DLibrary(), LessonInputWidth == 0.");
        }
        if (LessonInputHeight == 0)
        {
          throw std::invalid_argument("cnn::engine::complex::Lesson2DLibrary::Lesson2DLibrary(), LessonInputHeight == 0.");
        }
        if (LessonInputCount == 0)
        {
          throw std::invalid_argument("cnn::engine::complex::Lesson2DLibrary::Lesson2DLibrary(), LessonInputCount == 0.");
        }
        if (LessonOutputSize == 0)
        {
          throw std::invalid_argument("cnn::engine::complex::Lesson2DLibrary::Lesson2DLibrary(), LessonOutputSize == 0.");
        }
      }

      template <typename T>
      size_t Lesson2DLibrary<T>::GetLessonInputWidth() const
      {
        return LessonInputWidth;
      }

      template <typename T>
      size_t Lesson2DLibrary<T>::GetLessonInputHeight() const
      {
        return LessonInputHeight;
      }

      template <typename T>
      size_t Lesson2DLibrary<T>::GetLessonInputCount() const
      {
        return LessonInputCount;
      }

      template <typename T>
      size_t Lesson2DLibrary<T>::GetLessonOutputSize() const
      {
        return LessonOutputSize;
      }

      template <typename T>
      size_t Lesson2DLibrary<T>::GetLessonCount() const
      {
        return Lessons.size();
      }

      template <typename T>
      void Lesson2DLibrary<T>::PushBack()
      {
        typename ILesson2D<T>::Uptr newLesson = std::make_unique<Lesson2D<T>>(LessonInputWidth,
                                                                              LessonInputHeight,
                                                                              LessonInputCount,
                                                                              LessonOutputSize);
        Lessons.push_back(std::move(newLesson));
      }

      template <typename T>
      const ILesson2D<T>& Lesson2DLibrary<T>::GetLesson(const size_t index) const
      {
        if (index >= Lessons.size())
        {
          throw std::range_error("cnn::engine::complex::Lesson2DLibrary::GetLesson() const, index >= Lessons.size().");
        }
        return *(Lessons[index]);
      }

      template <typename T>
      ILesson2D<T>& Lesson2DLibrary<T>::GetLesson(const size_t index)
      {
        if (index >= Lessons.size())
        {
          throw std::range_error("cnn::engine::complex::Lesson2DLibrary::GetLesson(), index >= Lessons.size().");
        }
        return *(Lessons[index]);
      }

      template <typename T>
      const ILesson2D<T>& Lesson2DLibrary<T>::GetLastLesson() const
      {
        if (Lessons.size() == 0)
        {
          throw std::logic_error("cnn::engine::complex::Lesson2DLibrary::GetLastLesson(), Lessons.size() == 0.");
        }
        return *(Lessons.front());
      }

      template <typename T>
      ILesson2D<T>& Lesson2DLibrary<T>::GetLastLesson()
      {
        if (Lessons.size() == 0)
        {
          throw std::logic_error("cnn::engine::complex::Lesson2DLibrary::GetLastLesson(), Lessons.size() == 0.");
        }
        return *(Lessons.front());
      }
    }
  }
}