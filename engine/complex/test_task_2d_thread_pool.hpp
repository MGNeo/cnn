#pragma once

#include "i_test_task_2d_thread_pool.hpp"
#include "test_task_2d_thread.hpp"

#include <list>
#include <thread>
#include <sstream>

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

        std::list<typename ITestTask2DThread<T>::Uptr> Threads;

      };

      template <typename T>
      TestTask2DThreadPool<T>::TestTask2DThreadPool(ITestTask2DPool<T>& taskPool)
      {
        const int count = (std::thread::hardware_concurrency() > 0) ? std::thread::hardware_concurrency() : 1;
        for (int i = 0; i < count; ++i)
        {
          auto thread = std::make_unique<TestTask2DThread<T>>(taskPool);
          Threads.push_front(std::move(thread));
        }
      }

      template <typename T>
      void TestTask2DThreadPool<T>::Wait()
      {
        std::stringstream exceptionDescriptions;
        for (auto& thread : Threads)
        {
          try
          {
            thread->Wait();
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
    }
  }
}