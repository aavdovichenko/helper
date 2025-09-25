#pragma once

#include <condition_variable>
#include <functional>
#include <thread>

namespace Helper
{

class ThreadPool
{
public:
  ThreadPool(int threadCount = -1);
  ~ThreadPool();

  int getThreadCount() const;

  bool addJob(const std::function<void()>& job);
  void waitJobs();

private:
  ThreadPool(const ThreadPool& other) = delete;
  ThreadPool& operator=(const ThreadPool& other) = delete;

protected:
  struct Thread
  {
    std::thread m_thread;
    Thread* m_next = nullptr;
    ThreadPool* m_threadPool = nullptr;
    std::condition_variable m_condition;
    std::function<void()> m_job;

    void execute();
  };

  Thread* m_threads = nullptr;
  Thread* m_freeThreads = nullptr;
  int m_threadCount = 0;
  int m_freeThreadCount = 0;
  bool m_destroying = false;
  std::mutex m_mutex;
  std::condition_variable m_condition;
};

}
