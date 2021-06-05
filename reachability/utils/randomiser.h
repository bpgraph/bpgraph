/*************************************************************************
    > File Name: randomiser.h
    > Author: ZhangHeng
    > Mail: heng200888@163.com 
*/

#ifndef RANDOMISER_H_
#define RANDOMISER_H_
#include <chrono>
#include <list>
#include <random>

class randomiser
{
public:
	randomiser()
	{
	}
	~randomiser()
	{
	}
	static void uniform_dist(std::pair<uint, uint>& edge_pair, int bins, double probability)
	{

	}
	static size_t geometric_dist(double edge_probabiltiy){

	}
	
	/*
	 * Generate a probability in [0, 1]
	*/
	static double get_probability()
	{
		std::mt19937 rng;
		rng.seed(std::random_device()());
		std::uniform_real_distribution<> dist(0, std::nextafter(1, std::numeric_limits<double>::max())); // distribution in range [0, 1]
		return dist(rng);
	}

	/**
	 * Returns a Vertex id from a given vector of vertices
	 */
	static uint get_random_vertex_from_vector(std::vector<uint> v_list)
	{
		// Checks if size of vector == 1
		if (v_list.size() == 1) 
		{
			return v_list[0];
		}
		std::default_random_engine generator;
		std::uniform_int_distribution<size_t> distribution(1, v_list.size());
		return v_list[distribution(generator)-1];
	}
};




#endif

