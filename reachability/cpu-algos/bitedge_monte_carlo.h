
#ifndef BITEDGE_MONTE_CARLO_H_
#define BITEDGE_MONTE_CARLO_H_

#include <set>
#include <queue>
#include <bitset>

#include "../ugraph_io/ugraph_structures.h"
#include "../utils/memory_monitor.h"
#include "../utils/convergence_helper.h"
#include "../ugraph_io/file_io.h"
#include "../utils/globals.h"
#include "../utils/bitmap.h"


namespace CPU_ALGOS{
struct EdgeBits_T{
	std::vector<bool> bits;
	EdgeBits_T():bits(0){}
};

int bitedge_traverse_run(
	uint source, 
	uint target, 
	std::vector<initial_vertex> graph, 
	std::vector<EdgeBits_T> edgeBits,
	uint nEdges,
	int k);
//void find_k_monte_carlo(Graph & graph)
void find_k_bitedge_monte_carlo(std::vector<initial_vertex> & graph, uint nEdges, 
	std::vector<std::pair<uint, uint>> source_target_pairs, uint nQuery)
{
	std::cout << "Init Monte Carlo Sampling (Finding K)..." << std::endl;
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

	std::cout << std::endl << "Reading Source-Target file..." << std::endl;

	memory_monitor mm = memory_monitor();
	std::thread t1(&memory_monitor::update_peak_memory, std::ref(mm));
	t1.detach();
	

	while ( fabs(curr_avg_r - prev_avg_r) > ALGO_CONF::kReliabilityThreshold && 
		k < ALGO_CONF::kMaximumRound) { /*k < k_limit*/
		// Step up k
		k += ALGO_CONF::kKStepUp;
		std::cout << std::endl << "k = " << k << std::endl;
		int sumBits = k * nEdges;
		
		// Reset var
		reliability_k.clear();

		//Generate K sampling possible world
		auto start_g = std::chrono::high_resolution_clock::time_point::max();
		auto finish_g = std::chrono::high_resolution_clock::time_point::max();
		start_g = std::chrono::high_resolution_clock::now();
		int numVertices = graph.size();
		std::vector<EdgeBits_T> edgeBits(numVertices);

		int edgeInx = 0;
		for(int v = 0; v < numVertices; v++){
			uint vdegree = graph.at(v).nbrs.size();
			if( vdegree != 0 ){
				for( uint inbr = 0; inbr < vdegree; inbr++){
					edge ee = graph.at(v).nbrs.at(inbr).edgeValue;
					edgeInx ++;
					bool temp_bit;
					for(int i=0; i < k; i++){
						if ( check_exist( ee.probability.at(0)) ) {
							temp_bit = true;
						}else{
							temp_bit = false;
						}
						edgeBits.at(v).bits.push_back(temp_bit);
					}
				}
			}
		}
		finish_g = std::chrono::high_resolution_clock::now();
		auto duration_g = std::chrono::duration_cast<std::chrono::milliseconds>(finish_g - start_g).count();
		std::cout << "Sampling Possible World time = " << duration_g << " ms" << std::endl;


		for (size_t i = 0; i < source_target_pairs.size(); i++) {
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

				//Traverse K possible world
				auto start_t = std::chrono::high_resolution_clock::time_point::max();
				auto finish_t = std::chrono::high_resolution_clock::time_point::max();
				start_t = std::chrono::high_resolution_clock::now();
			#pragma omp parallel for num_threads(1)
				num_reached = num_reached + bitedge_traverse_run(source, target, graph, edgeBits, nEdges, k);
				finish_t = std::chrono::high_resolution_clock::now();
				auto duration_t = std::chrono::duration_cast<std::chrono::milliseconds>(finish_t - start_t).count();
				std::cout << "Traversal time = " << duration_t << " ms" << std::endl;
				std::cout<< "num_reached: "<< num_reached << std::endl;

				// Calculate reliability
				reliability = num_reached / (double)k;

				// Stop time
				finish = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();

				std::cout << "Current utilize : K=" << k << " possible world." << std::endl; 
				std::cout << "Reliability Estimator, R^ (" << source << ", " << target << ") = " << reliability << std::endl;
				std::cout << "Execution time = " << duration << " ms" << std::endl << std::endl;

				if (write_flag) {
					append_results_to_file(k, reliability, duration, mm.get_peak_memory(), 
						"MonteCarlo_k_" + std::to_string(i) + "_"+ std::to_string(numVertices)+ ".csv");
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
			append_results_to_file(k, diff_sq_sum / (ALGO_CONF::kRepeatForVariance - 1), 0, i, "MC_variance_" + std::to_string(numVertices)+".csv");
		}
		// Calulate avg r
		prev_avg_r = curr_avg_r;
		curr_avg_r = convergence_helper::get_avg_reliability(reliability_k);
	}
	mm.stop_monitoring();
}

void op_and (bool * a, bool* b, int k, bool *c){
	for(int i = 0; i< k ; i++)
		c[i] = a[i] & b[i];
}
void op_or (bool * a, bool* b, int k, bool *c){
	for(int i = 0; i < k ; i++)
		c[i] = a[i] | b[i];
}
int count_bits(bool * a, int k){
	int n = 0;
	for(int i = 0; i< k ; i++)
		if(a[i] == 1)
			n++;
	return n;
}

int bitedge_traverse_run(
	uint source, 
	uint target, 
	std::vector<initial_vertex> graph, 
	std::vector<EdgeBits_T> edgeBits,
	uint nEdges,
	int k)
{
	std::cout<< "Start source-target = " << source << ","<< target << std::endl;
	std::queue<uint> worklist;
	std::set<uint> explored;
	uint v, w;
	uint nVertices = graph.size();
	bool vbits[nVertices][k];
	for(int i = 0; i< nVertices; i++){
		for(int j = 0; j< k; j++){
			vbits[i][j] = false;
		}
	}

	// Add source in worklist
	for(int i = 0; i< k; i++)
		vbits[source][i] = true;
	worklist.push(source);
	explored.insert(source);
	if (source == target) {
		return 1*k;
	}
	while (!worklist.empty())
	{
		v = worklist.front();
		worklist.pop();
		// T -> S: Iterate through all ingoing edges from t -> s
		uint vdegree = graph.at(v).nbrs.size();
		if ( vdegree != 0) {
			for( uint i = 0; i < vdegree; i++){
				w = graph.at(v).nbrs.at(i).tgtIndex;
				edge ee = graph.at(v).nbrs.at(i).edgeValue;
				//bool * ee_bits = edgeBits.at(v).bits.at(i);
				std::vector<bool>::const_iterator bits_offset = edgeBits.at(v).bits.begin() + k*i;
				std::vector<bool>::const_iterator bits_offset2 = edgeBits.at(v).bits.begin() + k*(i+1);
				std::vector<bool> vee_bits( bits_offset, bits_offset2);

				bool ee_bits[k];

				for(int j = 0 ; j < k; j++)
					ee_bits[j] = vee_bits[j];
				bool temp_bits[k];
				op_and(ee_bits, vbits[v], k, temp_bits);
				op_or( vbits[w], temp_bits, k, temp_bits );
				for(int j = 0; j< k; j++)
					vbits[w][j] = temp_bits[j];
				if (explored.count(w) == 0) {
					worklist.push(w);
					explored.insert(w);
				}else {
					// Edge does not exist
				}
			}
		}
	}
	return count_bits(vbits[target], k);
}


}


#endif

