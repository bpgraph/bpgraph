/*************************************************************************
    > File Name: ugraph_structures.h
    > Author: ZhangHeng
    > Mail: zhanghenglab@gmail.com
*/

#ifndef UGRAPH_STRUCTURES_H
#define UGRAPH_STRUCTURES_H

#include <vector>
#include "../utils/randomiser.h"

#define bit_t unsigned char

struct vertex{
	//#ifdef WEIGHT
	//	float weight;
	//#endif
};

struct edge{
	// Still need to be optimized with index block
	std::vector<int> distance; // [0]: init 0
	std::vector<float> probability; // [0]: init raw prob
};

struct neighbor { 
	edge edgeValue; // dynamically change during iterations
	bool exist;
	uint tgtIndex;
};


class initial_vertex { // srcIndex -> nbrs (edgeValue, tgtIndex)
public:
	vertex vertexValue; // Null for no vertex probability and weight
	std::vector<neighbor> nbrs;
	initial_vertex():nbrs(0){}
	vertex& get_vertex_ref() {
		return vertexValue;
	}
};

class initial_path {
public:
	std::vector<uint> align_vertex;
	uint length;
	//std::vector<float> probability; // to distance=1
	initial_path():align_vertex(0), length(0){}//, probability(0){}
};



bool check_exist(double probability)
{
	if (randomiser::get_probability() < probability) {
		return true;
	}
	return false;
}


inline void completeEntry(
		const int src_vertex_index,	// Source vertex index.
		const int dst_vertex_index,	// Destination vertex index.
		edge* edge_address,	// Init edge probability
		vertex& src_vertex_ref,	// Pointer to the source vertex.
		vertex& dst_vertex_ref,  // Pointer to the destination vertex.
		float prob
		) {
//	src_vertex_ref.distance = ( src_vertex_index != arbparam ) ? INF : 0;
//	dst_vertex_ref.distance = ( dst_vertex_index != arbparam ) ? INF : 0;
	edge_address->distance.resize(2);
	edge_address->probability.resize(2);
	edge_address->distance[0] = edge_address->distance[1] = 1;
	edge_address->probability[0] = edge_address->probability[1] = prob;
}

inline void print_vertex_output(
		const uint vertexIndex,
		const vertex resultVertex,
		std::ofstream& outFile
		) {
	//	outFile << vertexIndex << ":\t" << resultVertex.weight << "\n";
}



#endif
