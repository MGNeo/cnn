#pragma once

#include <vector>

#include "Lesson2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      /*
      template <typename T>
      class Lesson2DLibrary
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Lesson2DLibrary(const Lesson2DTopology& lessonTopology = {});

        Lesson2DLibrary(const Lesson2DLibrary& library);

        Lesson2DLibrary(Lesson2DLibrary&& library) noexcept;

        Lesson2DLibrary& operator=(const Lesson2DLibrary& library);

        Lesson2DLibrary& operator=(Lesson2DLibrary&& library) noexcept;

        size_t GetLessonCount() const noexcept;

        void PushLesson(const Lesson2D& lesson);

        const Lesson2D<T>& GetLesson(const size_t index) const;

        const Lesson2DProtectingReference<T> GetLesson(const size_t index);

        void Clear() noexcept;

        void Reset() noexcept;

        void Save(std::ostream& ostream) const;

        void Load(std::istream& istream);

      private:

        Lesson2DTopology Topology;
        std::vector<Lesson2D<T>> Lessons;

      };
      */
    }
  }
}