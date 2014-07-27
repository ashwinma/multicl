#include "papi_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <papi.h>
#define MAX_PAPI_EVENTS	5
static int papi_events[MAX_PAPI_EVENTS] = {
			/*PAPI_FML_INS,
			PAPI_FAD_INS,
			PAPI_FDV_INS,
			PAPI_FSQ_INS,
			PAPI_FNV_INS,*/
			PAPI_FP_OPS, 
			PAPI_SP_OPS, 
			PAPI_DP_OPS, 
			PAPI_VEC_SP, 
			PAPI_VEC_DP
			/*
			PAPI_L1_DCM, 
			PAPI_L2_DCM, 
			PAPI_LD_INS,
			PAPI_SR_INS,
			PAPI_L1_LDM,
			PAPI_L1_STM,
			PAPI_L1_TCM,
			PAPI_L1_ICM*/
			};

static long long papi_values[MAX_PAPI_EVENTS] = {0};
static int papi_event_set = PAPI_NULL;

void papi_handle_error(int retval) {
	if(retval != PAPI_OK)
	{
		printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
		exit(1);
	}
}

void papi_init()
{
   /* Initialize the PAPI library */
   int retval = PAPI_library_init(PAPI_VER_CURRENT);
   if (retval != PAPI_VER_CURRENT)
   {
	   printf("PAPI library init error!\n");
	   exit(1);
   }

   printf("PAPI library init!\n");
   /* Enable and initialize multiplex support */
   papi_handle_error(PAPI_multiplex_init());
   printf("PAPI 1!\n");
   /* Create the Event Set */
   papi_handle_error(PAPI_create_eventset(&papi_event_set));
   printf("PAPI 2!\n");
   /* Convert the ''papi_event_set'' to a multiplexed event set */
   papi_handle_error(PAPI_assign_eventset_component(papi_event_set, 0));
   retval = PAPI_set_multiplex(papi_event_set);
   if ((retval == PAPI_EINVAL) && (PAPI_get_multiplex(papi_event_set) > 0))
   	printf("This event set already has multiplexing enabled\n");
   else if (retval != PAPI_OK) 
   	papi_handle_error(retval);
   printf("PAPI 3!\n");
   papi_handle_error(PAPI_add_events(papi_event_set, papi_events,
   					MAX_PAPI_EVENTS));
   printf("PAPI add events!\n");
}

long long papi_get_counter_val(int event_id)
{
	//char str[PAPI_MAX_STR_LEN];
	int i;
	for(i = 0; i < MAX_PAPI_EVENTS; i++)
	{
		if(event_id == papi_events[i])
		{
			//papi_handle_error(PAPI_event_code_to_name(event_id, str));
			return papi_values[i];
		}
	}
	fprintf(stderr, "PAPI Counter Not Initialized!\n");
	exit(1);
}

void papi_start_all_events()
{
	int status = 0;
	papi_handle_error(PAPI_state(papi_event_set, &status));
	if(!(status & PAPI_RUNNING))
		papi_handle_error(PAPI_start(papi_event_set));
}

void papi_accum_all_events()
{
	papi_handle_error(PAPI_accum(papi_event_set, papi_values));
}

void papi_stop_all_events()
{
	int status = 0;
	papi_handle_error(PAPI_state(papi_event_set, &status));
	if(!(status & PAPI_STOPPED))
		papi_handle_error(PAPI_stop(papi_event_set, papi_values));
}
	   
void papi_reset_all_events()
{
	papi_handle_error(PAPI_reset(papi_event_set));
}

void papi_print_all_events()
{
	int i;
	char str[PAPI_MAX_STR_LEN];
	for(i = 0; i < MAX_PAPI_EVENTS; i++)
	{
		papi_handle_error(PAPI_event_code_to_name(papi_events[i], str));
		printf("[PAPI] %s : %lld\n", str, papi_values[i]);
	}
//	double l1_hit_rate = 1 - ((double)papi_get_counter_val(PAPI_L1_DCM) / (papi_get_counter_val(PAPI_LD_INS) + papi_get_counter_val(PAPI_SR_INS)));
//	double l2_hit_rate = 1 - ((double)papi_get_counter_val(PAPI_L2_DCM) / (papi_get_counter_val(PAPI_LD_INS) + papi_get_counter_val(PAPI_SR_INS)));
//	printf("L1 Hit Rate: %g\n",  l1_hit_rate); 
//	printf("L2 Hit Rate: %g\n",  l2_hit_rate); 
}
