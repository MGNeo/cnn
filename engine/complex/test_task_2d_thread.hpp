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

        TestTask2DThread(typename ITestTask2DPool<T>& taskPool,
                         std::atomic<bool>& wrongFlag);

        ~TestTask2DThread();

      private:
        
        std::atomic<bool> StopCommand;
        ITestTask2DPool<T>& TaskPool;
        std::atomic<bool>& WrongFlag;
        std::future<void> Future;


        void Thread() const;
        
      };

      template <typename T>
      TestTask2DThread<T>::TestTask2DThread(typename ITestTask2DPool<T>& taskPool,
                                            std::atomic<bool>& wrongFlag)
        :
        StopCommand{ false },
        TaskPool{ taskPool },
        WrongFlag{ wrongFlag },
        Future{ std::async(std::launch::async, &TestTask2DThread::Thread, this) }
      {
      }

      template <typename T>
      void TestTask2DThread<T>::Thread() const
      {
        // This exception processing is not good, but others is worse.
        std::string exceptionDescription;
        try
        {
          while (StopCommand.load() == false)
          {
            auto task = TaskPool.Pop();
            if (task == nullptr)
            {
              break;
            }
            task->Execute();
          }
        }
        catch (std::exception& e)
        {
          exceptionDescription = "cnn::engine::complex::TestTask2DThread::Thread(), " + std::string(e.what());
        }
        catch (...)
        {
          exceptionDescription = "cnn::engine::complex::TestTask2DThread::Thread(), unknown exception has been caught.";
        }
        // If an exception will be thrown from here... then rest in peace.
        if (exceptionDescription.size() != 0)
        {
          static std::mutex mutex;
          std::lock_guard lock{ mutex };
          std::cerr << exceptionDescription << std::endl;
          WrongFlag.store(true, std::memory_order_relaxed);
        }
      }

      template <typename T>
      TestTask2DThread<T>::~TestTask2DThread()
      {
        StopCommand.store(true);
        // If the future has an exception we must softly notify the user.
        std::string exceptionDescription;
        try
        {
          Future.get();
        }
        catch (std::exception& e)
        {
          exceptionDescription = "cnn::engine::complex::TestTask2DThread::~TestTask2DThread(), " + std::string(e.what());
        }
        catch (...)
        {
          exceptionDescription = "cnn::engine::complex::TestTask2DThread::~TestTask2DThread(), unknown exception has been caught.";
        }
        // If an exception will be thrown from here... then rest in peace.
        if (exceptionDescription.size() != 0)
        {
          static std::mutex mutex;
          std::lock_guard lock{ mutex };
          std::cerr << exceptionDescription << std::endl;
          WrongFlag.store(true, std::memory_order_relaxed);
        }
      }
    }
  }
}