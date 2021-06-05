/*************************************************************************
    > File Name: file_io.h
    > Author: ZhangHeng
    > Mail: zhanghenglab@gmail.com
*/


#ifndef FILE_IO_H_
#define FILE_IO_H_

#include <string>
#include <cstring>
#include <cstdlib>
#include <stdio.h>
#include <iostream>


#include "ugraph_structures.h"
#include "../utils/globals.h"


uint read_graph(
		std::ifstream& inFile,
		const bool nondirectedGraph,
		std::vector<initial_vertex>& initGraph) {

	std::string line;
	char delim[3] = " \t";
	char* pch;
	uint nEdges = 0;

	// Read the input graph line-by-line.
	while( std::getline( inFile, line ) ) {
		if( line[0] < '0' || line[0] > '9' )
			continue;
		char cstrLine[256];
		std::strcpy( cstrLine, line.c_str() );
		uint firstIndex, secondIndex;
		float thirdIndex;

		pch = strtok(cstrLine, delim);
		if( pch != NULL )
			firstIndex = atoi( pch );
		else
			continue;
		pch = strtok( NULL, delim );
		if( pch != NULL )
			secondIndex = atoi( pch );
		else
			continue;
		pch = strtok( NULL, delim );
		if( pch != NULL )
			thirdIndex = atof( pch );
		else
			continue;

		uint theMax = std::max( firstIndex, secondIndex );
		uint srcVertexIndex = firstIndex;
		uint dstVertexIndex = secondIndex;
		float edgeProb = thirdIndex;
		if( initGraph.size() <= theMax )
			initGraph.resize(theMax+1);

		{
			neighbor nbrToAdd;
			nbrToAdd.tgtIndex = dstVertexIndex;
			nbrToAdd.exist = false;

			completeEntry(	srcVertexIndex,
							dstVertexIndex,
							&(nbrToAdd.edgeValue),
							(initGraph.at(srcVertexIndex).vertexValue),
							(initGraph.at(dstVertexIndex).vertexValue),
							edgeProb);

			initGraph.at(srcVertexIndex).nbrs.push_back( nbrToAdd );
			nEdges++;
		}
		if( nondirectedGraph ) {
			uint tmp = srcVertexIndex;
			srcVertexIndex = dstVertexIndex;
			dstVertexIndex = tmp;

			neighbor nbrToAdd;
			nbrToAdd.tgtIndex = srcVertexIndex;
			nbrToAdd.exist = false;

			completeEntry(	srcVertexIndex,
							dstVertexIndex,
							&(nbrToAdd.edgeValue),
							initGraph.at(srcVertexIndex).vertexValue,
							initGraph.at(dstVertexIndex).vertexValue,
							edgeProb);

			initGraph.at(dstVertexIndex).nbrs.push_back( nbrToAdd );
			nEdges++;
		}
		//printf("The edge is %d, %d, %f\n", srcVertexIndex, dstVertexIndex, edgeProb);
	}

	uint initialNumVertices = initGraph.size();
	printf("The number of vertices is %d\n", initialNumVertices);
	if( ( initialNumVertices % WARP_SIZE ) != 0 )
		initGraph.resize( ( ( initialNumVertices / WARP_SIZE ) + 1 ) * WARP_SIZE );

	return nEdges;

}


uint read_stlist(
		std::ifstream& inFile,
		std::vector<std::pair<uint, uint> >& stPairs) {

	std::string line;
	char delim[3] = " \t";
	char* pch;
	uint nQuery = 0;

	// Read the input graph line-by-line.
	while( std::getline( inFile, line ) ) {
		if( line[0] < '0' || line[0] > '9' )
			continue;
		char cstrLine[256];
		std::strcpy( cstrLine, line.c_str() );
		uint firstIndex, secondIndex;
		float thirdIndex;

		pch = strtok(cstrLine, delim);
		if( pch != NULL )
			firstIndex = atoi( pch );
		else
			continue;
		pch = strtok( NULL, delim );
		if( pch != NULL )
			secondIndex = atoi( pch );
		else
			continue;

		uint srcVertexIndex = firstIndex;
		uint targetVertexIndex = secondIndex;
		std::pair<uint, uint> stPair;
		stPair = std::make_pair(srcVertexIndex, targetVertexIndex);
		stPairs.push_back( stPair );
		nQuery++;
	}
	printf("The source-target query pairs are: \n" );
	for( auto i = 0; i < stPairs.size(); i++)
		printf("\t %d -> %d \n", stPairs.at(i).first, stPairs.at(i).second); 

	printf("The number of concurrent query is %d\n", nQuery);
	return nQuery;
}


void append_results_to_file(size_t k, double reliability, long long time, size_t memory, std::string file_name)
{
	std::ofstream file;
	file.open(file_name, std::ios_base::app);
	file << k << "," << reliability << "," << time << "," << memory << std::endl;
}


#endif
