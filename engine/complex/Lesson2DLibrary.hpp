#pragma once

#include <vector>

#include "Lesson2D.hpp"
#include "Lesson2DProtectingReference.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class Lesson2DLibrary
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Lesson2DLibrary() = default;

        Lesson2DLibrary(const Lesson2DLibrary& library) = default;

        Lesson2DLibrary(Lesson2DLibrary&& library) noexcept = default;

        Lesson2DLibrary& operator=(const Lesson2DLibrary& library);

        Lesson2DLibrary& operator=(Lesson2DLibrary&& library) noexcept = default;

        size_t GetLessonCount() const noexcept;

        // Exception guarantee: strong for this.
        void PushBack(const Lesson2D<T>& lesson);

        // Exception guarantee: strong for this.
        const Lesson2D<T>& GetLesson(const size_t index) const;

        // Exception guarantee: strong for this.
        const Lesson2DProtectingReference<T> GetLesson(const size_t index);

        // Clear the library from all lessons.
        void Clear() noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

        // Exception guarantee: strong for this and base for istream.
        // It loads full state.
        void Load(std::istream& istream);

      private:

        std::vector<Lesson2D<T>> Lessons;

      };

      template <typename T>
      Lesson2DLibrary<T>& Lesson2DLibrary<T>::operator=(const Lesson2DLibrary& library)
      {
        if (this != &library)
        {
          Lesson2DLibrary<T> tmpLibrary{ library };
          // Beware, it is very intimate place for strong exception guarantee.
          std::swap(*this, library);
        }
        return *this;
      }

      template <typename T>
      size_t Lesson2DLibrary<T>::GetLessonCount() const noexcept
      {
        return Lessons.size();
      }

      template <typename T>
      void Lesson2DLibrary<T>::PushBack(const Lesson2D<T>& lesson)
      {
        if (Lessons.size() != 0)
        {
          if (lesson.GetTopology() != Lessons.front().GetTopology())
          {
            throw std::invalid_argument("cnn::engine::complex::Lesson2DLibrary::PushBack(), lesson.GetTopology() != Lessons.front().GetTopology().");
          }
        }
        Lessons.push_back(lesson);
      }

      template <typename T>
      const Lesson2D<T>& Lesson2DLibrary<T>::GetLesson(const size_t index) const
      {
        if (index >= Lessons.size())
        {
          throw std::range_error("cnn::engine::complex::Lesson2DLibrary::GetLesson() const, index >= Lessons.size().");
        }
        return Lessons[index];
      }

      template <typename T>
      const Lesson2DProtectingReference<T> Lesson2DLibrary<T>::GetLesson(const size_t index)
      {
        if (index >= Lessons.size())
        {
          throw std::range_error("cnn::engine::complex::Lesson2DLibrary::GetLesson(), index >= Lessons.size().");
        }
        return Lessons[index];
      }

      template <typename T>
      void Lesson2DLibrary<T>::Clear() noexcept
      {
        Lessons.clear();
      }

      template <typename T>
      void Lesson2DLibrary<T>::Save(std::ostream& ostream) const
      {
        if (ostream.good() == false)
        {
          throw std::invalid_argument("cnn::engine:::complex::Lesson2DLibrary::Save(), ostream.good() == false.");
        }

        const size_t count = Lessons.size();
        ostream.write(reinterpret_cast<const char *const>(&count), sizeof(count));

        for (const auto& lesson : Lessons)
        {
          lesson.Save(ostream);
        }

        if (ostream.good() == false)
        {
          throw std::runtime_error("cnn::engine::complex::Lesson2DLibrary::Save(), ostream.good() == false.");
        }
      }

      template <typename T>
      void Lesson2DLibrary<T>::Load(std::istream& istream)
      {
        if (istream.good() == false)
        {
          throw std::invalid_argument("cnn::engine::complex::Lesson2DLibrary::Load(), istream.good() == false.");
        }

        size_t count{};
        decltype(Lessons) lessons;

        istream.read(reinterpret_cast<char* const>(&count), sizeof(count));

        for (size_t i = 0; i < count; ++i)
        {
          Lesson2D<T> lesson;
          lesson.Load(istream);
          if (i != 0)
          {
            if (lesson.GetTopology() != lessons.front().GetTopology())
            {
              throw std::logic_error("cnn::engine::complex::Lesson2DLibrary::Load(), lesson.GetTopology() != lessons.front().GetTopology().");
            }
          }
          lessons.push_back(std::move(lesson));
        }

        if (istream.good() == false)
        {
          throw std::runtime_error("cnn::engine::complex::Lesson2DLibrary::Load(), istream.good() == false.");
        }

        Lessons = std::move(lessons);
      }
    }
  }
}