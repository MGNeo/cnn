#pragma once

#include "i_test_task_2d_pool.hpp"

#include <list>
#include <mutex>

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class TestTask2DPool : public ITestTask2DPool<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        void Push(typename ITestTask2D<T>::Uptr&& task);

        typename ITestTask2D<T>::Uptr Pop() override;

        bool IsEmpty() const override;

      public:

        std::list<typename ITestTask2D<T>::Uptr> Tasks;
        mutable std::mutex Mutex;

      };

      template <typename T>
      void TestTask2DPool<T>::Push(typename ITestTask2D<T>::Uptr&& task)
      {
        if (task == nullptr)
        {
          throw std::invalid_argument("cnn::engine::complex::TestTaskPool2D::Push(), task == nullptr.");
        }
        std::lock_guard lock(Mutex);
        Tasks.push_front(std::move(task));
      }

      template <typename T>
      typename ITestTask2D<T>::Uptr TestTask2DPool<T>::Pop()
      {
        std::lock_guard lock(Mutex);
        if (Tasks.size() != 0)
        {
          auto task = std::move(Tasks.back());
          Tasks.pop_back();
          return task;
        } else {
          return nullptr;
        }
      }

      template <typename T>
      bool TestTask2DPool<T>::IsEmpty() const
      {
        std::lock_guard lock(Mutex);
        if (Tasks.size() == 0)
        {
          return true;
        } else {
          return false;
        }
      }

    }
  }
}