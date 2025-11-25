#pragma once

#include <cassert>
#include <condition_variable>
#include <functional>
#include <thread>

namespace Helper
{

class ThreadPool
{
public:
  typedef std::function<void(const void*)> CommonThreadFunction;

  inline ThreadPool(int threadCount = -1);
  inline ~ThreadPool();

  inline int getThreadCount() const;

  inline void setCommonThreadFunction(const CommonThreadFunction& f);
  inline bool addJob(const void* threadData);
  inline bool addJobs(const void* const * start, const void* const * end);
  inline bool addJob(const std::function<void()>& job);
  inline void waitJobs();

private:
  inline ThreadPool(const ThreadPool& other) = delete;
  inline ThreadPool& operator=(const ThreadPool& other) = delete;

  struct Thread
  {
    std::thread m_thread;
    Thread* m_next = nullptr;
    ThreadPool* m_threadPool = nullptr;
    std::condition_variable m_condition;
    std::function<void()> m_job;
    const void* m_data = nullptr;

    inline void execute();
  };

  inline Thread* lockThread(std::unique_lock<std::mutex>& lock);

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

// implementation

inline ThreadPool::ThreadPool(int threadCount) : m_threadCount(threadCount)
{
  if (m_threadCount < 0)
    m_threadCount = std::thread::hardware_concurrency();

  if (m_threadCount > 1)
  {
    m_threads = new Thread[m_threadCount];
    for (int i = 0; i < m_threadCount; i++)
    {
      m_threads[i].m_threadPool = this;
      m_threads[i].m_thread = std::thread([this, i]() { m_threads[i].execute(); });
    }
/*
    std::unique_lock lock(m_mutex);
    for (; m_freeThreadCount < m_threadCount;)
      m_condition.wait(lock);
*/
  }
}

inline ThreadPool::~ThreadPool()
{
  if (m_threads)
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    m_destroying = true;
    for (Thread* thread = m_freeThreads; thread; thread = thread->m_next)
      thread->m_condition.notify_one();
    lock.unlock();

    for (int i = 0; i < m_threadCount; i++)
      m_threads[i].m_thread.join();

    delete[] m_threads;
  }
}

inline int ThreadPool::getThreadCount() const
{
  return m_threadCount;
}

void ThreadPool::setCommonThreadFunction(const CommonThreadFunction& f)
{
  waitJobs();
  m_commonThreadFunction = f;
}

inline ThreadPool::Thread* ThreadPool::lockThread(std::unique_lock<std::mutex>& lock)
{
  for (; !m_freeThreads;)
    m_condition.wait(lock);

  Thread* thread = m_freeThreads;
  m_freeThreads = thread->m_next;
  m_freeThreadCount--;
  return thread;
}

inline bool ThreadPool::addJob(const void* threadData)
{
  assert(m_commonThreadFunction);
  if (!m_commonThreadFunction)
    return false;

  if (!m_threads)
  {
    m_commonThreadFunction(threadData);
    return true;
  }

  std::unique_lock<std::mutex> lock(m_mutex);
  Thread* thread = lockThread(lock);

  thread->m_job = nullptr;
  thread->m_data = threadData;
  thread->m_condition.notify_one();

  return true;
}

inline bool ThreadPool::addJobs(const void* const* start, const void* const* end)
{
  assert(m_commonThreadFunction);
  if (!m_commonThreadFunction)
    return false;

  if (!m_threads)
  {
    for (; start != end; ++start)
      m_commonThreadFunction(*start);
    return true;
  }

  std::unique_lock<std::mutex> lock(m_mutex);
  for (; start != end; ++start)
  {
    Thread* thread = lockThread(lock);
    thread->m_job = nullptr;
    thread->m_data = *start;
    thread->m_condition.notify_one();
  }

  return true;
}

inline bool ThreadPool::addJob(const std::function<void()>& job)
{
  if (!m_threads)
  {
    job();
    return true;
  }

  std::unique_lock<std::mutex> lock(m_mutex);
  Thread* thread = lockThread(lock);

  thread->m_job = job;
  thread->m_condition.notify_one();

  return true;
}

inline void ThreadPool::waitJobs()
{
  if (!m_threads)
    return;

  std::unique_lock<std::mutex> lock(m_mutex);
  for (; m_freeThreadCount < m_threadCount;)
    m_condition.wait(lock);
}

inline void ThreadPool::Thread::execute()
{
  std::unique_lock<std::mutex> lock(m_threadPool->m_mutex);
  for (; !m_threadPool->m_destroying;)
  {
    lock.unlock();

    if (m_job)
      m_job();
    else if (m_threadPool->m_commonThreadFunction)
    {
      assert(m_data);
      m_threadPool->m_commonThreadFunction(m_data);
    }

    lock.lock();
    m_job = nullptr;
    m_next = m_threadPool->m_freeThreads;
    m_threadPool->m_freeThreads = this;
    m_threadPool->m_freeThreadCount++;

    if (m_threadPool->m_destroying)
      break;

    m_threadPool->m_condition.notify_one();
    m_condition.wait(lock);
  }
}

}
