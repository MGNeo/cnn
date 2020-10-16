#pragma once

#include "i_test_task_2d_thread_pool.hpp"
#include "test_task_2d_thread.hpp"

#include <list>
#include <thread>

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class TestTask2DThreadPool : public ITestTask2DThreadPool<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        TestTask2DThreadPool(ITestTask2DPool<T>& taskPool);

        bool IsWrong() const override;

      private:

        // It can be changed only from any thread from Threads from false to true, otherwise the behavior is undefined.
        std::atomic<bool> WrongFlag;

        std::list<typename ITestTask2DThread<T>::Uptr> Threads;

      };

      template <typename T>
      TestTask2DThreadPool<T>::TestTask2DThreadPool(ITestTask2DPool<T>& taskPool)
      {
        const int count = (std::thread::hardware_concurrency() > 0) ? std::thread::hardware_concurrency() : 1;
        for (int i = 0; i < count; ++i)
        {
          auto thread = std::make_unique<TestTask2DThread<T>>(taskPool, WrongFlag);
          Threads.push_front(std::move(thread));
        }
      }

      template <typename T>
      bool TestTask2DThreadPool<T>::IsWrong() const
      {
        return WrongFlag.load(std::memory_order_relaxed);
      }
    }
  }
}