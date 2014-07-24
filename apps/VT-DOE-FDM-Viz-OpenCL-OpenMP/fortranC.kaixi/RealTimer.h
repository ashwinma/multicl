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

//#include <iostream>
#include <stdio.h>
#include <unistd.h>
//#include "mpi.h"
#include <sys/time.h>
//#include <string>

/// \class RealTimer RealTimer.h "Global/RealTimer.h"
///
/// \brief Compute elapsed time.
typedef struct RealTimer RealTimer;
  void Init(RealTimer *rt);
  /// Start the timer.
  void Start(RealTimer *rt);
  
  /// Stop the timer.
  void Stop(RealTimer *rt);

  /// Reset all counters to 0.
  void Reset(RealTimer *rt);
  
  /// Return the total time during which the timer was running.
  double Elapsed(RealTimer *rt);

  /// Number of times Stop() is called.
  int Count(RealTimer *rt);
  
  void Pause(RealTimer *rt);
  void Resume(RealTimer *rt);

  /// Print the current state of the counter to the stream.
  void Print(RealTimer *rt/*std::ostream& o*/, const char *desc);
  void PrintIndividual(RealTimer *rt, const char *desc);

  /// Return the system time.
  double CurrentTime(RealTimer *rt);

  /// Return true if the timer is running (started), otherwise return false (stopped).
  char IsRunning(RealTimer *rt);

struct RealTimer
{
#if 0
  /// Constructor.  If desc is provides, it will be output along with
  /// the RealTimer state when the RealTimer object is destroyed.
  //RealTimer(const char *desc="");
  //~RealTimer();

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
  //void Print(std::ostream& o) const;

  /// Return the system time.
  double CurrentTime() const;

  /// Return true if the timer is running (started), otherwise return false (stopped).
  bool IsRunning();
#endif
  char *fDesc;
  int time_offset;
  double start_time;
  double elapsed;
  char isRunning;
  int count;
};

/// Timer stream insertion operator
//inline std::ostream& operator<<(std::ostream& os, const RealTimer& t)
//{
//  t.Print(os);
//  return os;
//}

#if 0
inline RealTimer::RealTimer(const char *desc)
  : fDesc(desc),
    time_offset(0),
    start_time(0),
    elapsed(0.0),
    isRunning(false),
    count(0) {}
#endif

inline void Init(RealTimer *rt)
{
    rt->fDesc = "";
    rt->time_offset = 0;
    rt->start_time = 0;
    rt->elapsed = 0.0;
    rt->isRunning = 0;
    rt->count = 0;
	struct timeval tv;
	gettimeofday(&tv, NULL);
	double curTime = (double)(tv.tv_sec-(rt->time_offset)) + (double)tv.tv_usec*1e-6;
	rt->time_offset=(int)curTime;
	//time_offset=(int)CurrentTime();
}

#if 0
inline RealTimer::~RealTimer()
{
  if (fDesc != "")
    std::cout << "Timer " << fDesc << std::endl; 
}
#endif

inline void Start(RealTimer *rt)
{
  static const char *functionName = "RealTimer::Start()";
  if (rt->isRunning == 1) {
    fprintf(stdout, "%s: Warning: Timer has already been started.\n", functionName); 
  } else {
    rt->start_time = CurrentTime(rt);
    rt->isRunning = 1;
  }
}

inline void Stop(RealTimer *rt)
{
  static const char *functionName = "RealTimer::Stop()";
  if (rt->isRunning == 0) {
    fprintf(stdout, "%s: Warning: Timer has already been stopped.\n", functionName); 
  } else {
    rt->elapsed += CurrentTime(rt) - rt->start_time;
    rt->isRunning = 0;
    rt->count++;
  }
}

inline void PrintIndividual(RealTimer *rt, const char *desc)
{
  	fprintf(stdout, "[%s] Individual Stop Time: %g secs\n", desc, CurrentTime(rt) - rt->start_time);
}

inline char IsRunning(RealTimer *rt)
{
  return rt->isRunning;
}

inline void Reset(RealTimer *rt)
{
  rt->elapsed = 0.0;
  rt->start_time = 0;
  rt->count = 0;
}

inline double Elapsed(RealTimer *rt)
{
  static const char *functionName = "inline float Timer::Elapsed() const";
  if (rt->isRunning == 1) {
    fprintf(stdout, "%s: Warning: Timer is still running.\n", functionName); 
    return rt->elapsed + CurrentTime(rt) - rt->start_time;
  }
  return rt->elapsed;
}

inline int Count(RealTimer *rt) {return rt->count;}

inline double CurrentTime(RealTimer *rt)
{
// #ifdef SUN
//   timespec ts;
//   clock_gettime(CLOCK_REALTIME, &ts);
//   return (double)(ts.tv_sec-time_offset) + (double)ts.tv_nsec*1e-9;
//#else
#if 1
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)(tv.tv_sec-(rt->time_offset)) + (double)tv.tv_usec*1e-6;
#else
  /* MPI_Wtime gives a better resolution than gettimeofday */
  return MPI_Wtime()-(rt->time_offset);
#endif
//#endif
}

inline void Print(RealTimer *rt/*std::ostream& o*/, const char *desc) 
{
  if (rt->count > 1)
  fprintf(stdout, "[%s] %g secs %d times %g secs each\n", desc, 
  							rt->elapsed, rt->count, rt->elapsed/rt->count);
  else
  	fprintf(stdout, "[%s] %g secs %d times\n", desc, rt->elapsed, rt->count);
/*
  o << elapsed << " secs "
    << count << " times";
  if (count > 1)
    o << " " << (elapsed/count) << " secs each";
*/
}

#endif //  COUTIL_GLOBAL_REALTIMER
