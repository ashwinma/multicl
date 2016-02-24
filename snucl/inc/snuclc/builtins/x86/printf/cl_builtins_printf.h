

#ifndef _cl_builtins_printf_h_
#define _cl_builtins_printf_h_

#include <cl_cpu_types.h>
#include <stdlib.h>
#include <vector>

class Print {
  char* format;
  std::vector<void *> arglist;

  mutable int inx;

  public:
  Print(char* _format);

  template <typename T>
    Print& operator,(T arg) {
      T* _arg = (T *)malloc(sizeof(T));
      *_arg = arg;
      arglist.push_back(_arg);
      return *this;
    }

  Print& operator,(float arg);
  Print& operator,(int arg);
  Print& operator,(short arg);
  Print& operator,(char arg);
  
  void* get_arg() const;
  int print() const;

  ~Print();

};


int snu_printf(const Print& p);


#define printf(x, ...) snu_printf((Print(x), ##__VA_ARGS__))

#endif
