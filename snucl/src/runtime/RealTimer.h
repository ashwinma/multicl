//--------------------------------------------------------------------------
// File:    RealTimer.h
// Module:  Global
// Author:  Keith Bisset
// Created: January 28 1997
// Time-stamp: "2004-12-07 09:22:13 sxs"
// Description: Compute elapsed time and memory usage.
//
// @COPYRIGHT@
//
//--------------------------------------------------------------------------

#ifndef COUTIL_GLOBAL_REALTIMER
#define COUTIL_GLOBAL_REALTIMER

#include <iostream>
#include <unistd.h>
#include "mpi.h"
#include <sys/time.h>
#include <string>

namespace Global {
  
/// \class RealTimer RealTimer.h "Global/RealTimer.h"
///
/// \brief Compute elapsed time.
class RealTimer
{
public:
  /// Constructor.  If desc is provides, it will be output along with
  /// the RealTimer state when the RealTimer object is destroyed.
  RealTimer(const std::string& desc="");
  ~RealTimer();

  void Init();
  /// Start the timer.
  void Start();
  
  /// Stop the timer.
  void Stop();

  /// Reset all counters to 0.
  void Reset();
  
  /// Return the total time during which the timer was running.
  double Elapsed() const;

  /// Number of times Stop() is called.
  int Count() const;
  
  void Pause();
  void Resume();

  /// Print the current state of the counter to the stream.
  void Print(std::ostream& o) const;
  void PrintCurrent() const;
  double CurrentElapsed() const;

  /// Return the system time.
  double CurrentTime() const;

  /// Return true if the timer is running (started), otherwise return false (stopped).
  bool IsRunning();

private:
  std::string fDesc;
  int time_offset;
  double start_time;
  double elapsed;
  bool isRunning;
  int count;
};

/// Timer stream insertion operator
inline std::ostream& operator<<(std::ostream& os, const RealTimer& t)
{
  t.Print(os);
  return os;
}

inline RealTimer::RealTimer(const std::string& desc)
  : fDesc(desc),
    time_offset(0),
    start_time(0),
    elapsed(0.0),
    isRunning(false),
    count(0) {}

inline void RealTimer::Init()
{
	timeval tv;
	gettimeofday(&tv, NULL);
	double curTime = (double)(tv.tv_sec-time_offset) + (double)tv.tv_usec*1e-6;
	time_offset=(int)curTime;
	//time_offset=(int)CurrentTime();
}

inline RealTimer::~RealTimer()
{
#if 0
  if (fDesc != "")
    std::cout << "Timer " << fDesc << std::endl; 
#endif
}

inline void RealTimer::Start()
{
  static std::string functionName("RealTimer::Start()");
  if (isRunning == true) {
    std::cout << functionName << ": Warning: Timer " << fDesc << " has already been started." <<  std::endl; 
  } else {
    start_time = CurrentTime();
    isRunning = true;
  }
}

inline void RealTimer::Stop()
{
  static std::string functionName("RealTimer::Stop()");
  if (isRunning == false) {
    std::cout << functionName << ": Warning: Timer " << fDesc << " has already been stopped." <<  std::endl; 
  } else {
    elapsed += CurrentTime() - start_time;
    isRunning = false;
    count++;
  }
}

inline bool
RealTimer::IsRunning()
{
  return isRunning;
}

inline void RealTimer::Reset()
{
  elapsed = 0.0;
  start_time = 0;
  count = 0;
}

inline double RealTimer::Elapsed() const
{
  static std::string functionName("inline float Timer::Elapsed() const");
  if (isRunning == true) {
    std::cout << functionName << ": Warning: Timer " << fDesc << " is still running." <<  std::endl; 
    return elapsed + CurrentTime() - start_time;
  }
  return elapsed;
}

inline int RealTimer::Count() const {return count;}

inline double RealTimer::CurrentTime() const
{
// #ifdef SUN
//   timespec ts;
//   clock_gettime(CLOCK_REALTIME, &ts);
//   return (double)(ts.tv_sec-time_offset) + (double)ts.tv_nsec*1e-9;
//#else
#if 1
  timeval tv;
  gettimeofday(&tv, NULL);
  return (double)(tv.tv_sec-time_offset) + (double)tv.tv_usec*1e-6;
#else
  /* MPI_Wtime gives a better resolution than gettimeofday */
  return MPI_Wtime()-time_offset;
#endif
//#endif
}

inline double RealTimer::CurrentElapsed() const {
  static std::string functionName("inline float Timer::CurrentElapsed() const");
  if (isRunning == true) {
    std::cout << functionName << ": Warning: Timer " << fDesc << " is still running." <<  std::endl; 
  }
  return CurrentTime() - start_time;
}

inline void RealTimer::PrintCurrent() const
{
  std::cout << CurrentTime() - start_time << " secs" << std::endl;
}

inline void RealTimer::Print(std::ostream& o) const
{
  o << elapsed << " secs "
    << count << " times";
  if (count > 1)
    o << " " << (elapsed/count) << " secs each";
}

} // namespace
#endif //  COUTIL_GLOBAL_REALTIMER
