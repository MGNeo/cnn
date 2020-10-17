#pragma once

#include "i_test_task_2d_thread.hpp"
#include "i_test_task_2d_pool.hpp"

#include <future>
#include <thread>
#include <atomic>

namespace cnn
{
  namespace engine
  {
    namespace complex
    {
      template <typename T>
      class TestTask2DThread : public ITestTask2DThread<T>
      {

        static_assert(std::is_floating_point<T>::value);

      public:

        TestTask2DThread(typename ITestTask2DPool<T>& taskPool);

        void Wait() override;

        ~TestTask2DThread();

      private:

        ITestTask2DPool<T>& TaskPool;
        std::atomic<bool> StopCommand;
        std::future<void> Future;

        void Thread() const;

      };

      template <typename T>
      TestTask2DThread<T>::TestTask2DThread(typename ITestTask2DPool<T>& taskPool)
        :
        TaskPool{ taskPool },
        StopCommand{ true },
        Future{ std::async(std::launch::async, &TestTask2DThread::Thread, this) }
      {
      }

      template <typename T>
      void TestTask2DThread<T>::Wait()
      {
        Future.get();
      }

      template <typename T>
      TestTask2DThread<T>::~TestTask2DThread()
      {
        StopCommand.store(true);
      }

      template <typename T>
      void TestTask2DThread<T>::Thread() const
      {
        while (StopCommand.load() == false)
        {
          auto task = TaskPool.Pop();
          if (task != nullptr)
          {
            task->Execute();
          } else {
            return;
          }
        }
      }
    }
  }
}