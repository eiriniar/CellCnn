/*
 * my_logicle.h
 *
 *  Created on: Dec 5, 2010
 *      Author: jolly
 */

#ifndef MY_LOGICLE_H_
#define MY_LOGICLE_H_
#include "logicle.h"

void logicle_scale(double t, double w, double m, double a, double* x, int n);
class IllegalParameter {public: const char * message ();};
#endif /* MY_LOGICLE_H_ */
