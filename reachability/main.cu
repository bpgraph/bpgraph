#include <string>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>


//#include "utils/file_io.h"
#include "ugraph_io/ugraph_structures.h"
#include "cpu-algos/naive_monte_carlo.h"
#include "cpu-algos/bitedge_monte_carlo.h"
#include "cpu-algos/path_sampling_monte_carlo.h"
#include "core/prepare_uncertain_graph.cuh"

#include "core/reorganize_uncertain_graph.cuh"

#include "utils/cuda_error_check.cuh"
#include "core/mem_alloc_graph.cuh"
#include "core/process_path_graph.cuh"

#include "utils/cu_bitmap.h"

#include <mpi.h>


#define WARP_SIZE 32
#define WARP_SIZE_SHIFT 5

#define MPI_CHECK(call) \
    if((call) != MPI_SUCCESS) { \
        cerr << "MPI error calling \""#call"\"\n"; \
        my_abort(-1); }


#define GLOBAL_DIST_CONSTRAIN 3

int isGPUExists(void){
    int device_count, device;
    int gpu_device_count = 0;
    struct cudaDeviceProp properties;
    cudaError_t cuda_result_code = cudaGetDeviceCount(&device_count);
    if (cuda_result_code != cudaSuccess) device_count = 0;
    /* machines with no GPUs can still report one emulation device */
    for (device = 0; device < device_count; ++device) {
        cudaGetDeviceProperties(&properties, device);
        if (properties.major != 9999) { /* 9999 means emulation only */
            ++gpu_device_count;
        }
    }
    if (gpu_device_count) return 1;
    else return 0;
}

template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {

	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to access specified file: " + file_name + "\n" );
}


int main( int argc, char** argv ) {

	std::string usage =
	"\t Entry Command arguments:\n\
		-Input graph edge list: E.g., --edgelist in.txt\n\
		-Source-target list: E.g., --stlist st.txt\n\
	Additional arguments:\n\
		-Output file (default: out.txt). E.g., --output myout.txt\n\
		-Is the input graph directed (default:yes). To make it undirected: --undirected\n\
		-Device ID in single-GPU mode (default: 0). E.g., --device 1\n\
		-Number of GPUs (default: 1). E.g., --nDevices 2.\n";

	// Required variables for initialization.
	std::ifstream inputEdgeList;
	std::ifstream stList;
	std::ofstream outputFile;
	bool nonDirectedGraph = false;		// By default, the graph is directed.
	int num_device = 1;
	int single_device_id = 0;


	try{

		for( int iii = 1; iii < argc; ++iii )
			if( !strcmp( argv[iii], "--edgelist" ) && iii != argc-1) 
				openFileToAccess< std::ifstream >( inputEdgeList, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--stlist" ) && iii != argc-1)
				openFileToAccess< std::ifstream >( stList, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--output" ) && iii != argc-1)
				openFileToAccess< std::ofstream >( outputFile, std::string( argv[iii+1] ) );
			else if( !strcmp(argv[iii], "--undirected"))
				nonDirectedGraph = true;
			else if( !strcmp( argv[iii], "--nDevices" ) && iii != argc-1)
				num_device = std::atoi( argv[iii+1] );
			else if( !strcmp( argv[iii], "--device" ) && iii != argc-1)
				single_device_id = std::atoi( argv[iii+1] );

		if( !inputEdgeList.is_open() )
			throw std::runtime_error( "Initialization Error: The input edge list has not been specified." );
		if( !stList.is_open() )
			throw std::runtime_error( "Initialization Error: The source-target list has not been specified." );
		if( !outputFile.is_open() )
			openFileToAccess< std::ofstream >( outputFile, "temp_out.txt" );
	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n" << "Usage: " << usage << "\nExiting." << std::endl;
		return( EXIT_FAILURE );
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}

	try {
		if( num_device > 1 ) {
			std::cout << num_device << " devices will be processing the graph.\n";
		} else {
			std::cout << "The device with the ID " << single_device_id << " will process the graph" << std::endl;
		}

		std::vector<initial_vertex> inMemGraph(0);
		
		uint num_edge = read_graph(
				inputEdgeList,
				nonDirectedGraph,
				inMemGraph);
		uint num_vertex = inMemGraph.size();
		std::cout << "Input graph collected with " << num_vertex << " vertices and " << num_edge << " edges." << std::endl;

		std::vector<std::pair<uint, uint>> source_target_pairs(0);
		uint num_query = read_stlist(stList, source_target_pairs);

		auto start_cpu = std::chrono::high_resolution_clock::time_point::max();
		auto finish_cpu = std::chrono::high_resolution_clock::time_point::max();
		/***************************************************************
		 * Start to CPU computations
		***************************************************************/
		//1a. Entirety Sampling: Naive Monte Carlo Algo
std::cout << "########### 1a. k_naive_monte_carlo#######"<< std::endl;
		start_cpu = std::chrono::high_resolution_clock::now();
		//CPU_ALGOS::find_k_naive_monte_carlo(inMemGraph, num_edge, source_target_pairs, num_query);
		finish_cpu = std::chrono::high_resolution_clock::now();
		auto duration_cpu_naive_k = std::chrono::duration<double, std::milli> (finish_cpu - start_cpu).count();
std::cout << "#############k_naive_monte_carlo execution time: " << duration_cpu_naive_k << " ms" << std::endl << std::endl;

		//1b. Entirety Sampling: Monte Carlo BFS Algo
std::cout << "###########k_monte_carlo_bfs#######"<< std::endl;
		start_cpu = std::chrono::high_resolution_clock::now();
		//CPU_ALGOS::find_k_monte_carlo_bfs(inMemGraph, num_edge, source_target_pairs, num_query);
		finish_cpu = std::chrono::high_resolution_clock::now();
		auto duration_cpu_k_bfs = std::chrono::duration<double, std::milli> (finish_cpu - start_cpu).count();
std::cout << "#############k_bfs_monte_carlo execution time: " << duration_cpu_k_bfs << " ms"<< std::endl << std::endl;		//1c. Entirety Sampling: BitEdge Monte Carlo Algo
		//1c. Entirety Sampling: Monte Carlo Bitedge Algo
std::cout << "###########k_monte_carlo_bfs#######"<< std::endl;
		start_cpu = std::chrono::high_resolution_clock::now();
		//CPU_ALGOS::find_k_bitedge_monte_carlo(inMemGraph, num_edge, source_target_pairs, num_query);
		finish_cpu = std::chrono::high_resolution_clock::now();
		auto duration_cpu_k_bitedge = std::chrono::duration<double, std::milli> (finish_cpu - start_cpu).count();
std::cout << "#############k_bitedge_monte_carlo execution time: " << duration_cpu_k_bitedge << " ms"<< std::endl << std::endl;		//1c. Entirety Sampling: BitEdge Monte Carlo Algo
		
		//2a. Partition Sampling: ProbTree Algo
		//CPU_ALGOS::find_k_probtree_monte_carlo(inMemGraph, num_edge, stList);

		//3a. Path Sampling Algo: Single Core Path
		//CPU_ALGOS::find_k_path_single_monte_carlo(inMemGraph, num_edge, stList);

		//3b. Path Sampling Algo: Multi-Core Path (Openmp)
		//CPU_ALGOS::find_k_path_multi_monte_carlo(inMemGraph, num_edge, stList);


		std::cout << "======GPU Runtime ======= "  << std::endl;	
		std::vector<std::vector<uint> >  path_graph = bpgraph::reorganize_uncertain_graph(
			source_target_pairs,
			inMemGraph,
			num_edge,
			GLOBAL_DIST_CONSTRAIN);
		//Print identified paths
		int path_finding = path_graph.size();
		std::cout<< " Identify paths: "<< std::endl;
		for( int i = 0; i < path_finding; i++){
			std::cout << "Path #" << i+1 << ": ";
			for(int j = 0; j < (path_graph[i]).size(); j++) {
				std::cout << path_graph[i][j] << " ";
			}
			std::cout << std::endl;
		}


		std::vector<uint> indices_range( num_device + 1 );
		indices_range.at(0) = 0;
		indices_range.at( indices_range.size() - 1 ) = inMemGraph.size();
		if( num_device > 1 ){
			uint approxmiateNumEdgesPerDevice = num_edge / num_device;
			for( unsigned int dev = 1; dev < num_device; ++dev ) {
				unsigned int accumulatedEdges = 0;
				uint movingVertexIndex = indices_range.at( dev - 1 );
				while( accumulatedEdges < approxmiateNumEdgesPerDevice ) {
					accumulatedEdges += inMemGraph.at( movingVertexIndex ).nbrs.size();
					++movingVertexIndex;
				}
				movingVertexIndex &= ~( WARP_SIZE - 1 );
				indices_range.at( dev ) = movingVertexIndex;
			}
		}

		/* Check multiple device availability */
		for( uint dev_id = 0; dev_id < num_device; ++dev_id ) {
			CUDAErrorCheck( cudaSetDevice( ( num_device == 1 ) ? single_device_id : dev_id ) );
			CUDAErrorCheck( cudaFree( 0 ) );
		}

		/* Init device buffers */
		uint * d_vertex_offset;
		uint * d_edge_offset;
		float * d_edge_probability;

		uint * d_edge_in_path;
		uint * d_edge_id;

		CUDAErrorCheck( cudaMalloc(&d_vertex_offset, (num_vertex+1) * sizeof(uint)));
		CUDAErrorCheck( cudaMalloc(&d_edge_offset, num_edge * sizeof(uint)));
		CUDAErrorCheck( cudaMalloc(&d_edge_probability, num_edge * sizeof(float)));
		CUDAErrorCheck( cudaMalloc(&d_edge_in_path, path_graph.size() * sizeof(uint)));
		CUDAErrorCheck( cudaMalloc(&d_edge_id, num_edge * sizeof(uint)));
		

		/* Allocation of path graph for uncertain graph datasets*/
		if (num_device == 1) {
			bpgraph::mem_alloc_graph(
				&inMemGraph,
				num_edge,
				path_graph,
				d_vertex_offset,
				d_edge_offset,
				d_edge_probability,
				d_edge_in_path,
				d_edge_id,
				single_device_id);
						/* Alloc GPU st-pairs */
		} else { 
			// bpgraph::mem_alloc_distribute_graph(
			// 	&inMemGraph,
			// 	num_edge,
			// 	&path_graph,
			// 	&indices_range,
			// 	num_device);
		}

		for( uint dev_id = 0; dev_id < num_device; ++dev_id ) {
			CUDAErrorCheck( cudaSetDevice( ( num_device == 1 ) ? single_device_id : dev_id ) );
			CUDAErrorCheck( cudaDeviceSynchronize() );
		}

		

		
//		/********************************
//		 * Only source-to-target rechability evalution found in graphs.
//		 ********************************/
		if (num_device == 1) {
			bpgraph::reacheability_single_path_process(
				d_vertex_offset,
				d_edge_offset,
				d_edge_probability,
				d_edge_in_path,
				d_edge_id,
				num_vertex,
				num_edge,
				path_graph.size(),
				source_target_pairs,
				single_device_id);
		} else {
			// bpgraph::reacheability_single_distribute_path_process(
			// 	d_vertex_offset,
			// 	d_edge_offset,
			// 	d_edge_probability,
			// 	d_edge_in_path,
			// 	d_edge_id,
			// 	&indices_range,
			// 	num_device);
		}

		//	/********************************
//		 * Only source-to-target rechability evalution found in graphs.
//		 ********************************/
		// if (num_device == 1) {
		// 	bpgraph::distance_single_path_process(
		// 		d_vertex_offset,
		// 		d_edge_offset,
		// 		d_edge_probability,
		// 		d_edge_in_path,
		// 		d_edge_id,
		// 		num_vertex,
		// 		num_edge,
		// 		path_graph.size(),
		// 		d_sources,
		// 		d_targets,
		// 		num_query,
		// 		single_device_id);
		// } else {
		// 	bpgraph::distance_single_distribute_path_process(
		// 		d_vertex_offset,
		// 		d_edge_offset,
		// 		d_edge_probability,
		// 		d_edge_in_path,
		// 		d_edge_id,
		// 		&indices_range,
		// 		num_device);
		// }




//
//
//		/********************************
//		 * Execution the graph with Any path (WCC).
//		 ********************************/
//
//		any_path_process(
//				&inMemGraph,
//				num_edge,
//				outputFile,
//				&indicesRange,
//				num_device,
//				singleDeviceID,
//				commMethod );
//
//		/********************************
//		 * Execution the graph with All path (WCC).
//		 ********************************/
//
//		all_path_process(
//				&inMemGraph,
//				num_edge,
//				outputFile,
//				&indicesRange,
//				num_device,
//				singleDeviceID,
//				commMethod );


/*
int data[] = {1,2,3,4,5,8};
    size_t length = 6;
    std::cout<<"raw data"<<std::endl;
    for (size_t i = 0 ;i < length ;i++)
    {
        std::cout<<data[i]<<" ";
    }
    std::cout<<std::endl;
    cu_bitmap *bbb = new cu_bitmap(10);
    for (size_t i = 0 ;i < length ;i++)
    {
        bbb->set_bit(data[i]);
    }
    std::cout<<"\nbitmap processed\n";
    for (size_t i = 0 ; i <= 10;i++)
    {
        size_t ret = bbb->get_bit(i);
        std::cout<<i<<":"<<ret << " ";
    }
	*/


		std::cout << "Done." << std::endl;
		return( EXIT_SUCCESS );
	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n" << "Exiting." << std::endl;
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}


}
