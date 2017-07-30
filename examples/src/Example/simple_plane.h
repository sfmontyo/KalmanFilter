#ifndef SIMPLE_PLANE_H
#define SIMPLE_PLANE_H

#include "kalman/ekfilter.hpp"


class cSimplePlaneEKF : public Kalman::EKFilter<double,1> {
public:
	cSimplePlaneEKF();

protected:

	void makeA();
	void makeH();
	void makeV();
	void makeR();
	void makeW();
	void makeQ();
	void makeProcess();
	void makeMeasure();

	double Period, Mass, Bfriction, Portance, Gravity;
};

typedef cSimplePlaneEKF::Vector SimpleVector;
typedef cSimplePlaneEKF::Matrix SimpleMatrix;

#endif
