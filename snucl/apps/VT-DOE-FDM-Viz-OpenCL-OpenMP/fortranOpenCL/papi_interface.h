#ifndef __PAPI_FDM__
#define __PAPI_FDM__
void papi_handle_error(int retval);
void papi_init();
long long papi_get_counter_val(int event_id);
void papi_start_all_events();
void papi_accum_all_events();
void papi_stop_all_events();
void papi_reset_all_events();
void papi_print_all_events();

#endif //__PAPI_FDM__
