#pragma once

#include "i_test_task_2d_thread_pool.hpp"

#include <list>
#include <future>
#include <sstream>
#include <atomic>
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

        void Wait() override;

      private:

        // This function is static because the function must not have access to Futures.
        static void Thread(ITestTask2DPool<T>& taskPool, std::atomic<bool>& sharedStopCommand);

        std::atomic<bool> StopCommand;

        std::list<std::future<void>> Futures;

      };

      template <typename T>
      TestTask2DThreadPool<T>::TestTask2DThreadPool(ITestTask2DPool<T>& taskPool)
        :
        StopCommand{ false }
      {
        try
        {
          const int count = (std::thread::hardware_concurrency() > 0) ? std::thread::hardware_concurrency() : 1;
          for (int i = 0; i < count; ++i)
          {
            std::future<void> future = std::async(std::launch::async, TestTask2DThreadPool<T>::Thread, std::ref(taskPool), std::ref(StopCommand));
            Futures.push_back(std::move(future));
          }
        }
        catch (...)
        {
          // If the constructor is failed then we must send stop signal all already started threads.
          StopCommand.store(true);
          throw;
        }
      }

      template <typename T>
      void TestTask2DThreadPool<T>::Wait()
      {
        std::stringstream exceptionDescriptions;
        for (auto& future : Futures)
        {
          try
          {
            future.get();
          }
          catch (std::exception& e)
          {
            exceptionDescriptions << "cnn::engine::complex::TestTask2DThreadPool::Wait(), " << e.what() << std::endl;
          }
          catch (...)
          {
            exceptionDescriptions << "cnn::engine::Complex::TestTask2DThreadPool::Wait(), unknown exception has been caught." << std::endl;
          }
        }
        if (exceptionDescriptions.str().size() != 0)
        {
          throw std::runtime_error(exceptionDescriptions.str());
        }
      }

      template <typename T>
      void TestTask2DThreadPool<T>::Thread(ITestTask2DPool<T>& taskPool, std::atomic<bool>& sharedStopCommand)
      {
        try
        {
          while (sharedStopCommand.load() == false)
          {
            auto task = taskPool.Pop();
            if (task != nullptr)
            {
              task->Execute();
            } else {
              break;
            }
          }
        }
        catch (...)
        {
          // If the thread is failed then we must send stop signal all threads.
          sharedStopCommand.store(true);
          throw;
        }
      }
    }
  }
}