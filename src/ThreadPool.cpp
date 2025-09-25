#include <Helper/ThreadPool.h>

namespace Helper
{

ThreadPool::ThreadPool(int threadCount) : m_threadCount(threadCount)
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

ThreadPool::~ThreadPool()
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

int ThreadPool::getThreadCount() const
{
  return m_threadCount;
}

bool ThreadPool::addJob(const std::function<void()>& job)
{
  if (!m_threads)
  {
    job();
    return true;
  }

  std::unique_lock<std::mutex> lock(m_mutex);
  for (; !m_freeThreads;)
    m_condition.wait(lock);

  Thread* thread = m_freeThreads;
  m_freeThreads = thread->m_next;
  m_freeThreadCount--;
  thread->m_job = job;
  thread->m_condition.notify_one();

  return true;
}

void ThreadPool::waitJobs()
{
  if (!m_threads)
    return;

  std::unique_lock<std::mutex> lock(m_mutex);
  for (; m_freeThreadCount < m_threadCount;)
    m_condition.wait(lock);
}

void ThreadPool::Thread::execute()
{
  std::unique_lock<std::mutex> lock(m_threadPool->m_mutex);
  for (;!m_threadPool->m_destroying;)
  {
    lock.unlock();

    if (m_job)
      m_job();

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
