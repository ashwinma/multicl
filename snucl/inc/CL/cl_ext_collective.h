#ifndef __CL_EXT_COLLECTIVE_H
#define __CL_EXT_COLLECTIVE_H

#ifdef __cplusplus
extern "C" {
#endif

#define CL_DOUBLE                              0x10DF

#define CL_COMMAND_ALLTOALL_BUFFER             0x1300
#define CL_COMMAND_BROADCAST_BUFFER            0x1301
#define CL_COMMAND_SCATTER_BUFFER              0x1302
#define CL_COMMAND_GATHER_BUFFER               0x1303
#define CL_COMMAND_ALLGATHER_BUFFER            0x1304
#define CL_COMMAND_REDUCE_BUFFER               0x1305
#define CL_COMMAND_ALLREDUCE_BUFFER            0x1306
#define CL_COMMAND_REDUCESCATTER_BUFFER        0x1307
#define CL_COMMAND_SCAN_BUFFER                 0x1308

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueScanBuffer(cl_command_queue * cmd_queue_list,
                    cl_uint num_buffers,
                    cl_mem * src_buffer_list,
                    cl_mem * dst_buffer_list,
                    size_t * src_offset,
                    size_t * dst_offset,
                    size_t cb,
                    cl_channel_type datatype,
                    cl_uint num_events_in_wait_list,
                    const cl_event * event_wait_list,
                    cl_event * event_list);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReduceScatterBuffer(cl_command_queue * cmd_queue_list,
                             cl_uint num_buffers,
                             cl_mem * src_buffer_list,
                             cl_mem * dst_buffer_list,
                             size_t * src_offset,
                             size_t * dst_offset,
                             size_t cb,
                             cl_channel_type datatype,
                             cl_uint num_events_in_wait_list,
                             const cl_event * event_wait_list,
                             cl_event * event_list);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueAllReduceBuffer(cl_command_queue * cmd_queue_list,
                         cl_uint num_buffers,
                         cl_mem * src_buffer_list,
                         cl_mem * dst_buffer_list,
                         size_t * src_offset,
                         size_t * dst_offset,
                         size_t cb,
                         cl_channel_type datatype,
                         cl_uint num_events_in_wait_list,
                         const cl_event * event_wait_list,
                         cl_event * event_list);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReduceBuffer(cl_command_queue cmd_queue,
                      cl_uint num_src_buffers,
                      cl_mem * src_buffer_list,
                      cl_mem dst_buffer,
                      size_t * src_offset,
                      size_t dst_offset,
                      size_t cb,
                      cl_channel_type datatype,
                      cl_uint num_events_in_wait_list,
                      const cl_event * event_wait_list,
                      cl_event * event);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueAlltoAllBuffer(cl_command_queue * cmd_queue_list,
                        cl_uint num_buffers,
                        cl_mem * src_buffer_list,
                        cl_mem * dst_buffer_list,
                        size_t * src_offset,
                        size_t * dst_offset,
                        size_t cb,
                        cl_uint num_events_in_wait_list,
                        const cl_event * event_wait_list,
                        cl_event * event_list);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueAlltoAllVBuffer(cl_command_queue * cmd_queue_list,
                         cl_uint num_buffers,
                         cl_mem * src_buffer_list,
                         cl_mem * dst_buffer_list,
                         size_t * src_offset,
                         size_t * dst_offset,
                         size_t * cb,
                         cl_uint num_events_in_wait_list,
                         const cl_event * event_wait_list,
                         cl_event * event_list);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueBroadcastBuffer(cl_command_queue * cmd_queue_list,
                         cl_mem src_buffer,
                         cl_uint num_dst_buffers,
                         cl_mem * dst_buffer_list,
                         size_t src_offset,
                         size_t * dst_offset,
                         size_t cb,
                         cl_uint num_events_in_wait_list,
                         const cl_event * event_wait_list,
                         cl_event * event_list);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueScatterBuffer(cl_command_queue * cmd_queue_list,
                       cl_mem src_buffer,
                       cl_uint num_dst_buffers,
                       cl_mem * dst_buffer_list,
                       size_t src_offset,
                       size_t * dst_offset,
                       size_t cb, // size sent to each dst buffer
                       cl_uint num_events_in_wait_list,
                       const cl_event * event_wait_list,
                       cl_event * event_list);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueGatherBuffer(cl_command_queue cmd_queue,
                      cl_uint num_src_buffers,
                      cl_mem * src_buffer_list,
                      cl_mem dst_buffer,
                      size_t * src_offset,
                      size_t dst_offset,
                      size_t cb, // size for any single receive
                      cl_uint num_events_in_wait_list,
                      const cl_event * event_wait_list,
                      cl_event * event);

extern CL_API_ENTRY cl_int CL_API_CALL
clEnqueueAllGatherBuffer(cl_command_queue * cmd_queue_list,
                         cl_uint num_buffers,
                         cl_mem * src_buffer_list,
                         cl_mem * dst_buffer_list,
                         size_t * src_offset,
                         size_t * dst_offset,
                         size_t cb,
                         cl_uint num_events_in_wait_list,
                         const cl_event * event_wait_list,
                         cl_event * event_list);

#ifdef __cplusplus
}
#endif


#endif /* __CL_EXT_COLLECTIVE_H */
