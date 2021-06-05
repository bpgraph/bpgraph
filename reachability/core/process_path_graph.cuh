#ifndef PROCESS_PATH_GRAPH_CUH
#define PROCESS_PATH_GRAPH_CUH

#include "../ugraph_io/ugraph_structures.h"
#include "api.cuh"
#include "../utils/cuda_error_check.cuh"
#include "../utils/memory_monitor.h"
#include "../utils/convergence_helper.h"
#include "../ugraph_io/file_io.h"
#include "../utils/globals.h"

#include "curand_kernel.h" 


#include <cooperative_groups.h>

using namespace cooperative_groups;
namespace cg = cooperative_groups;

#define BLOCK_SIZE COMPILE_TIME_DETERMINED_BLOCK_SIZE
const int BLOCK_QUEUE_SIZE = 128;
const int SUB_QUEUE_LEN = 32;
const int NUM_SUB_QUEUES = 4;

namespace bpgraph
{

__global__ void 
initialize_empty_path (
    uint * d_vertex_offset,
	uint * d_edge_offset,
	float * d_edge_probability,
	uint * d_edge_in_path,
	uint * d_edge_id,
    uint num_vertex,
    uint num_edge,
    uint num_path,
    bit_t *d_path_bit) {
        const size_t NUM_TOTAL_BIT = sizeof(bit_t) * (((num_edge)>>3) + 1); 
        // Generate bits along paths, each thread fetch one path for sampling

        //for edge e in p.get_edges(): 
         //       expand_path(p, e)
    } 



__global__ void 
single_path_child_kernel(
    uint * d_vertex_offset,
	uint * d_edge_offset,
	float * d_edge_probability,
	uint * d_edge_in_path,
	uint * d_edge_id,
    uint num_vertex,
    uint num_edge,
    uint num_path) {
        


        /*const uint path_id = tidWithinCTA + blockIdx.x * blockDim.x;
	    if( path_id >= num_path )
		    return;

        initialize_empty_path(
		    Path* local_p,	
		    Path* p	);

        expand_path(
                volatile Path* local_p,	
                Edge* e	) ;

        
        reduce_vertex(
            Path p[], 
            Vertex v)*/
        
    }


__device__ float target_reduce_sum(thread_group g, float *temp, float val)
{
    int lane = g.thread_rank();

    for (int i = g.size() / 2; i > 0; i /= 2)
    {
        temp[lane] = val;
        g.sync(); 
        if(lane<i) val += temp[lane + i];
        g.sync(); 
    }
    return val;
}

__device__ float target_thread_sum(float *input, int n)
{
    float sum = 0;

    for(int i = blockIdx.x * blockDim.x + threadIdx.x;
        i < n / 4; 
        i += blockDim.x * gridDim.x)
    {
        float4 in = ((float4*)input)[i];
        sum += in.x + in.y + in.z + in.w;
    }
    return sum;
}

__global__ void sum_kernel_block(float *sum , float *input, int n)
{
    float my_sum = target_thread_sum(input, n);

    extern __shared__ float temp[];
    auto g = this_thread_block();
    float block_sum = target_reduce_sum(g, temp, my_sum);
   // printf("Thread block size: %d, rank: %d.\n", g.size(), g.thread_rank());

    if (g.thread_rank() == 0)
        atomicAdd(sum, block_sum);

}

__device__ bool check_edge_exists(int tid, float prob_value){
    long clock_for_rand = clock();
    curandState state;
    curand_init(clock_for_rand, tid, 0, &state);
    float dev_random = abs(curand_uniform( &state ));
    //printf("Debug %d thread generate random: %f - %f\n", tid, dev_random, prob_value);
    return (dev_random < prob_value);
}


__global__ void single_path_child_kernel_block(int target, int * d_reached, 
            int depth, uint *d_vertex_offset, uint num_vertex, 
            uint *d_edge_offset, float * d_edge_probability, int *d_distance, 
            int *d_currQ, int currQSize,
            int *d_next_queue, int *d_next_queue_size) {

        /*
        * allocate device global & shared memory variable
        */
        __shared__ int shared_next_queue[BLOCK_QUEUE_SIZE];
        __shared__ int shared_next_queue_size, share_block_global_queue_Idx;

        //Current thread id
        int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
        if (thread_id == 0)
            shared_next_queue_size = 0;
        if (thread_id > num_vertex)
            return;
        __syncthreads();
        if (thread_id < currQSize) {
            const int parent = d_currQ[thread_id];
            for (int i=d_vertex_offset[parent]; 
                    i < d_vertex_offset[parent+1]; i++) {
                
                uint child = d_edge_offset[i];
                if (check_edge_exists (thread_id, d_edge_probability[child]) ){ 
                    if(child == target) {
                        (*d_reached) = 1;
                        // printf("debug: %d, %d-%d, %d\n", thread_id, parent, child, target);
                    }
                        
                    if (atomicMin(&d_distance[child], INT_MAX) == -1) {
                        d_distance[child] = depth;
                        const int sharedQIdx = atomicAdd(&shared_next_queue_size, 1);

                        if (sharedQIdx < BLOCK_QUEUE_SIZE) {
                            shared_next_queue[sharedQIdx] = child;
                            // printf("debug3: %d %d, %d\n", BLOCK_QUEUE_SIZE, child, *d_reached);
                        }
                        else { 
                            shared_next_queue_size = BLOCK_QUEUE_SIZE;
                            const int global_queue_Idx = atomicAdd(d_next_queue_size, 1);
                            d_next_queue[global_queue_Idx] = child;
                            // printf("debug3: %d %d, %d\n", BLOCK_QUEUE_SIZE, child, *d_reached);
                        }
                    }
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) //offset for global memory
            share_block_global_queue_Idx = atomicAdd(d_next_queue_size, shared_next_queue_size);
        __syncthreads();
        for (int i=threadIdx.x; i<shared_next_queue_size; i+=blockDim.x) {// fill the global memory
            d_next_queue[share_block_global_queue_Idx + i] = shared_next_queue[i];
        }
    }


int single_path_child_execute(
    uint source, uint target,
    uint * d_vertex_offset,
    uint * d_edge_offset,
    float * d_edge_probability,    
    uint num_vertex,
    uint num_edge) {
    //initialize data
    int *d_distance;
    int *d_parent;
    int *d_currQ;
    int *d_next_queue;
    int *d_next_queue_size;
    int currQSize{1};
    int depth = 0;
    int numBlocks;
    //execution
    std::vector<int> distanceCheck;
    distanceCheck.resize(num_vertex);
    std::fill(distanceCheck.begin(), distanceCheck.end(), -1);
    distanceCheck[source] = 0;
    CUDAErrorCheck( cudaMalloc(&d_distance, num_vertex * sizeof(int)));
    CUDAErrorCheck( cudaMemcpy(d_distance, distanceCheck.data(), num_vertex * sizeof(int), cudaMemcpyHostToDevice));
    // init medium data sets
    CUDAErrorCheck( cudaMalloc(&d_currQ, num_vertex * sizeof(int)));
    CUDAErrorCheck( cudaMalloc(&d_next_queue, num_vertex * sizeof(int)));
    CUDAErrorCheck( cudaMalloc(&d_next_queue_size, sizeof(int)));
    CUDAErrorCheck( cudaMemcpy(d_currQ, &source, sizeof(int), cudaMemcpyHostToDevice));
    CUDAErrorCheck( cudaMemset(d_next_queue_size, 0, sizeof(int)));
    auto start = std::chrono::high_resolution_clock::now();
    int *reached;
    CUDAErrorCheck( cudaHostAlloc( (void**)&reached, sizeof(int), cudaHostAllocPortable ) );
    (*reached) = 0;
    int *d_reached;
    CUDAErrorCheck( cudaMalloc(&d_reached, sizeof(int)));
    CUDAErrorCheck( cudaMemcpy(d_reached, reached, sizeof(int), cudaMemcpyHostToDevice) );
    while (currQSize && !(*reached) ) {
        numBlocks = ((currQSize - 1) / BLOCK_SIZE) + 1;
        single_path_child_kernel_block<<<numBlocks, BLOCK_SIZE>>> (
                target, reached,
                ++depth, d_vertex_offset, num_vertex, d_edge_offset, d_edge_probability, d_distance,
                d_currQ, currQSize, d_next_queue, d_next_queue_size);
        CUDAErrorCheck( cudaDeviceSynchronize() );
        CUDAErrorCheck( cudaMemcpyAsync(&currQSize, d_next_queue_size, sizeof(int), cudaMemcpyDeviceToHost) );
        CUDAErrorCheck( cudaMemcpyAsync(reached, d_reached, sizeof(int), cudaMemcpyDeviceToHost) );
        // std::cout << depth << "th it33: " << *reached << std::endl;
        if( *(reached) )
            break;
        CUDAErrorCheck( cudaMemcpyAsync(d_currQ, d_next_queue, sizeof(int) * currQSize, cudaMemcpyDeviceToDevice) );
        //CUDAErrorCheck( cudaMemset(d_next_queue_size, 0, sizeof(int)) );
        // std::cout << " Debug:" << depth << "th it44: " << *reached << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t = end - start;
    //obtained distances 
    CUDAErrorCheck( cudaMemcpy(distanceCheck.data(), d_distance, num_vertex * sizeof(int), cudaMemcpyDeviceToHost) );
    if(currQSize == 0)
        return 0;
    return 1;
}

uint 
reacheability_single_path_process(
    uint * d_vertex_offset,
	uint * d_edge_offset,
	float * d_edge_probability,
	uint * d_edge_in_path,
	uint * d_edge_id,
    uint num_vertex,
    uint num_edge,
    uint num_path,
    std::vector<std::pair<uint, uint>> source_target_pairs,
    const int single_device_id) {        
        int numBlocks = ((num_vertex - 1) / BLOCK_SIZE) + 1;
        std::cout<< "GPU block configuration: BLOCK_SIZE=" << BLOCK_SIZE << ", number of blocks="<< numBlocks << std::endl;
        // get_paths(d_edge_in_path, d_edge_id);

        bit_t *d_path_bit;
        const size_t NUM_TOTAL_BIT = sizeof(bit_t) * (((num_edge)>>3) + 1); 
        cudaMalloc((void **)&d_path_bit, NUM_TOTAL_BIT);
		cudaMemset(d_path_bit, 0, NUM_TOTAL_BIT);

        initialize_empty_path <<< numBlocks, BLOCK_SIZE >>> (
            d_vertex_offset,
	        d_edge_offset,
	        d_edge_probability,
            d_edge_in_path,
	        d_edge_id,
            num_vertex,
            num_edge,
            num_path,
            d_path_bit
        );


        int num_reached = 0;
        int k = 0;	
        double reliability;
        std::vector<double> reliability_k, reliability_j;
        double curr_avg_r = 2.0;
        double prev_avg_r = 3.0;
        double avg_r = 0.0;
        double diff_sq_sum = 0.0;
        bool write_flag = true;
        uint source, target;
        std::pair<uint, uint> source_target_pair;

        memory_monitor mm = memory_monitor();
        std::thread t1(&memory_monitor::update_peak_memory, std::ref(mm));
        t1.detach();

        while ( fabs(curr_avg_r - prev_avg_r) > ALGO_CONF::kReliabilityThreshold && k < ALGO_CONF::kMaximumRound) { /*k < k_limit*/
		// Step up k
		k += ALGO_CONF::kKStepUp;
		std::cout << "k = " << k << std::endl;

		// Reset var
		reliability_k.clear();
        uint * d_sources;
		uint * d_targets;
        uint num_query = source_target_pairs.size();
        bpgraph::mem_alloc_st_pairs(
            source_target_pairs,
            d_sources,
            d_targets,
            num_query,0);

		for (size_t i = 0; i < num_query; i++) {
			source_target_pair = source_target_pairs[i];
			source = source_target_pair.first;
			target = source_target_pair.second;

			// Reset var
			reliability_j.clear();
			diff_sq_sum = 0.0;
			write_flag = true;

			for (int j = 0; j < ALGO_CONF::kRepeatForVariance; j++) {
				std::cout << j << "th iteration" << std::endl;
				// Reset initial conditions
				num_reached = 0;

				// Start time
				auto start = std::chrono::high_resolution_clock::time_point::max();
				auto finish = std::chrono::high_resolution_clock::time_point::max();
				start = std::chrono::high_resolution_clock::now();
				mm.start_monitoring();

                #pragma omp parallel for num_threads(1)
				for (int i = 0; i < k; i++) { // kKStep for controling K sampling world
					//int is_find = monte_carlo_run(source, target, graph);
                    std::cout << "Find source -> target: "<< source << "-" << target << std::endl;
                    int path_is_find = single_path_child_execute(
                                source, 
                                target, 
                                d_vertex_offset,
                                d_edge_offset,
                                d_edge_probability,
                                num_vertex,
                                num_edge);
					num_reached = num_reached + path_is_find;
					if(path_is_find)
					{
						std::cout << "Source-target is reached in Possible World #" << i <<std::endl;
					}
					else
					{
						std::cout << "Not reached in Possible World #" << i <<std::endl;
					}
				}
				std::cout<< "Total num_reached: "<< num_reached << std::endl;

				// Calculate reliability
				reliability = num_reached / (double)k;

				// Stop time
				finish = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration<double, std::milli> (finish - start).count();

				std::cout << "Current utilize : K=" << k << " possible world." << std::endl; 
				std::cout << "Reliability Estimator, R^ (" << source << ", " << target << ") = " << reliability << std::endl;
				std::cout << "Execution time = " << duration << " ms" << std::endl << std::endl;

				if (write_flag) {
					append_results_to_file(k, reliability, duration, mm.get_peak_memory(), "MonteCarlo_k_" + std::to_string(i) + ".csv");
					write_flag = false;
				}
				// Add r to vector
				reliability_j.push_back(reliability);
			}

			// Add r to vector of r
			reliability_k.push_back(reliability);

			// Variance calculation
			avg_r = convergence_helper::get_avg_reliability(reliability_j);
			for (int j = 0; j < ALGO_CONF::kRepeatForVariance; j++) {
				auto difference_sq = pow(reliability_j[j] - avg_r, 2);
				diff_sq_sum += difference_sq;
			}
			append_results_to_file(k, diff_sq_sum / (ALGO_CONF::kRepeatForVariance - 1), 0, i, "MC_variance.csv");
		}
		// Calulate avg r
		prev_avg_r = curr_avg_r;
		curr_avg_r = convergence_helper::get_avg_reliability(reliability_k);
	}


        return num_reached;
    }



}
#endif