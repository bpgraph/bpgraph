/*************************************************************************
    > File Name: globals.h
    > Author: ZhangHeng
    > Mail: zhanghenglab@gmail.com
*/

#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#define WARP_SIZE 32
#define WARP_SIZE_SHIFT 5
#define COMPILE_TIME_DETERMINED_BLOCK_SIZE 128


namespace ALGO_CONF{
	const double kRecursiveSamplingThreshold = 5;
	const size_t kRSSr = 50;
	const size_t kRSSThreshold = 10;	// 10
	const int kMaximumRound = 5; // !!for test program, normally set to (1000-10000)
	const int kKStepUp = 5; // 250
	const int kRepeatForVariance = 1; //100
	const double kReliabilityThreshold = -1;//0.0001;
}

#endif
