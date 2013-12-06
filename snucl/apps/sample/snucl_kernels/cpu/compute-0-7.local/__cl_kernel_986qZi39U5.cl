__kernel void sample(__global int* dst, __global int* src, int offset) {
  int id = get_global_id(0);
  dst[id] = src[id] + offset;
}
