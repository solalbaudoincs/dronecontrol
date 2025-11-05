/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_quadcopter_model_api.h
 *
 * Code generation for function 'quadcopter_model'
 *
 */

#ifndef _CODER_QUADCOPTER_MODEL_API_H
#define _CODER_QUADCOPTER_MODEL_API_H

/* Include files */
#include "emlrt.h"
#include "mex.h"
#include "tmwtypes.h"
#include <string.h>

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
void quadcopter_model(real_T x[12], real_T u[4], real_T dxdt[12]);

void quadcopter_model_api(const mxArray *const prhs[2], const mxArray **plhs);

void quadcopter_model_atexit(void);

void quadcopter_model_initialize(void);

void quadcopter_model_terminate(void);

void quadcopter_model_xil_shutdown(void);

void quadcopter_model_xil_terminate(void);

#ifdef __cplusplus
}
#endif

#endif
/* End of code generation (_coder_quadcopter_model_api.h) */
