#pragma once

#include "Lesson2D.hpp"

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      // Lesson2DProtectingReference is a type which implements semantics of protecting reference to Lesson2D.
      // The smart reference proxies all methods of Lesson2D and doesn't allow to use methods, which change
      // the topology of the target lesson.
      // It allow to protect consistency of complex objects, which contain the target lesson as its part.
      template <typename T>
      class Lesson2DProtectingReference
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        Lesson2DProtectingReference(Lesson2D<T>& lesson) noexcept;

        Lesson2DProtectingReference(const Lesson2DProtectingReference& lessonReference) noexcept;

        Lesson2DProtectingReference(Lesson2DProtectingReference&& lessonReference) noexcept = delete;

        Lesson2DProtectingReference& operator=(const Lesson2DProtectingReference& lessonReference) noexcept = delete;

        Lesson2DProtectingReference& operator=(Lesson2DProtectingReference&& lessonReference) noexcept = delete;

        const Lesson2DTopology& GetTopology() const noexcept;

        const convolution::Map2D<T>& GetConstInput() const noexcept;

        convolution::Map2DProtectingReference<T> GetInput() const noexcept;

        const common::Map<T>& GetConstOutput() const noexcept;

        common::MapProtectingReference<T> GetOutput() const noexcept;

        // It clears the state without changing of the topology of the lesson.
        void Clear() const noexcept;

        // Exception guarantee: base for ostream.
        // It saves full state.
        void Save(std::ostream& ostream) const;

      private:

        Lesson2D<T>& Lesson;

      };

      template <typename T>
      Lesson2DProtectingReference<T>::Lesson2DProtectingReference(Lesson2D<T>& lesson) noexcept
        :
        Lesson{ lesson }
      {
      }

      template <typename T>
      Lesson2DProtectingReference<T>::Lesson2DProtectingReference(const Lesson2DProtectingReference& lessonReference) noexcept
        :
        Lesson{ lessonReference.Lesson }
      {
      }

      template <typename T>
      const Lesson2DTopology& Lesson2DProtectingReference<T>::GetTopology() const noexcept
      {
        return Lesson.GetTopology();
      }

      template <typename T>
      const convolution::Map2D<T>& Lesson2DProtectingReference<T>::GetConstInput() const noexcept
      {
        return Lesson.GetInput();
      }

      template <typename T>
      convolution::Map2DProtectingReference<T> Lesson2DProtectingReference<T>::GetInput() const noexcept
      {
        return Lesson.GetInput();
      }

      template <typename T>
      const common::Map<T>& Lesson2DProtectingReference<T>::GetConstOutput() const noexcept
      {
        return Lesson.GetOutput();
      }

      template <typename T>
      common::MapProtectingReference<T> Lesson2DProtectingReference<T>::GetOutput() const noexcept
      {
        return Lesson.GetOutput();
      }

      template <typename T>
      void Lesson2DProtectingReference<T>::Clear() const noexcept
      {
        Lesson.Clear();
      }

      template <typename T>
      void Lesson2DProtectingReference<T>::Save(std::ostream& ostream) const
      {
        Lesson.Save(ostream);
      }
    }
  }
}