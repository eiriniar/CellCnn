/*
 * my_logicle.cpp
 *
 *  Created on: Dec 5, 2010
 *      Author: jolly
 */

#include "my_logicle.h"

void logicle_scale(double t, double w, double m, double a, double* x, int n) {
	Logicle *l = new Logicle(t, w, m, a);
	//Logicle *l = (Logicle *)logicle_initialize(t, w, m, a, 0);
	for(int j=0;j<n;j++) {
		x[j] = l->scale(x[j]);
	}
	delete l;
}

