/*************************************************************************
    > File Name: convergence.h
    > Author: ZhangHeng
    > Mail: zhanghenglab@gmail.com
*/

#ifndef CONVERGENCE_HELPER_H_
#define CONVERGENCE_HELPER_H_


#include <vector>
#include <math.h>

namespace convergence_helper{

	double get_avg_reliability(std::vector<double> r)
	{
		double subtotal = 0.0;
		for (double r_i : r) {
			subtotal += r_i;
		}
		return subtotal / (double)r.size();
	}

	/*
	 * Returns std R for MC Sampling
	 */
	double get_std_reliability_MC(double r, std::vector<int> i_st)
	{
		double subtotal = 0.0;
		for (int i : i_st) {
			subtotal += pow((i - r), 2);
		}
		return sqrt(subtotal) / (double)i_st.size();
	}
};


#endif

