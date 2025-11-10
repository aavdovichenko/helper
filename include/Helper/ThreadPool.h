#pragma once

#include <condition_variable>
#include <functional>
#include <thread>
#include <vector>

namespace Helper
{

class ThreadPool
{
public:
  typedef std::function<void(const void*)> CommonThreadFunction;

  ThreadPool(int threadCount = -1);
  ~ThreadPool();

  int getThreadCount() const;

  void setCommonThreadFunction(const CommonThreadFunction& f);
  bool addJob(const void* threadData);
  bool addJobs(const void* const * start, const void* const * end);
  bool addJob(const std::function<void()>& job);
  void waitJobs();

private:
  ThreadPool(const ThreadPool& other) = delete;
  ThreadPool& operator=(const ThreadPool& other) = delete;

  struct Thread
  {
    std::thread m_thread;
    Thread* m_next = nullptr;
    ThreadPool* m_threadPool = nullptr;
    std::condition_variable m_condition;
    std::function<void()> m_job;
    const void* m_data = nullptr;

    void execute();
  };

  Thread* lockThread(std::unique_lock<std::mutex>& lock);

protected:

  Thread* m_threads = nullptr;
  Thread* m_freeThreads = nullptr;
  int m_threadCount = 0;
  int m_freeThreadCount = 0;
  bool m_destroying = false;
  CommonThreadFunction m_commonThreadFunction;
  std::mutex m_mutex;
  std::condition_variable m_condition;
};

}
