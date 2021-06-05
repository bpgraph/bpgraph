
#ifndef MONTE_CARLO_H_
#define MONTE_CARLO_H_

#include <queue>
#include <set>

#include "../ugraph_io/ugraph_structures.h"
#include "../utils/memory_monitor.h"
#include "../utils/convergence_helper.h"
#include "../ugraph_io/file_io.h"
#include "../utils/globals.h"



namespace CPU_ALGOS{

int traverse_run(uint source, uint target, std::vector<initial_vertex> graph);

//void find_k_monte_carlo(Graph & graph)
void find_k_naive_monte_carlo(
	std::vector<initial_vertex> & graph, 
	uint nEdges, 
	std::vector<std::pair<uint, uint>> source_target_pairs, 
	uint nQuery)
{
	std::cout << std::endl;
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
	memory_monitor mm = memory_monitor();
	std::thread t1(&memory_monitor::update_peak_memory, std::ref(mm));
	t1.detach();
	

	while ( fabs(curr_avg_r - prev_avg_r) > ALGO_CONF::kReliabilityThreshold 
		&& k < ALGO_CONF::kMaximumRound) /*k < k_limit*/
	{ 
		// Step up k
		k += ALGO_CONF::kKStepUp;
		std::cout << "k = " << k << std::endl;

		// Reset var
		reliability_k.clear();

		//Generate K sampling possible world
		auto start_g = std::chrono::high_resolution_clock::time_point::max();
		auto finish_g = std::chrono::high_resolution_clock::time_point::max();
		start_g = std::chrono::high_resolution_clock::now();
		std::vector<initial_vertex> graph_k[k];
		int numVertices = graph.size();
		for(int i=0; i < k; i++)
		{
			graph_k[i].resize(numVertices);
			int n_kEdges = 0;
			for(int v = 0; v < numVertices; v++)
			{
				uint vdegree = graph.at(v).nbrs.size();
				if( vdegree != 0 )
				{
					neighbor nbrToAdd;
					for( uint inbr = 0; inbr < vdegree; inbr++)
					{
						edge ee = graph.at(v).nbrs.at(inbr).edgeValue;
						if ( check_exist( ee.probability.at(0)) )
						{
							nbrToAdd.tgtIndex = graph.at(v).nbrs.at(inbr).tgtIndex;
							graph_k[i].at(v).nbrs.push_back( nbrToAdd );
							n_kEdges++;
						}
					}
				}
			}
			std::cout<< "Num of Edge in Graph " << i << "th is :" << n_kEdges << std::endl;
		}
		finish_g = std::chrono::high_resolution_clock::now();
		auto duration_g = std::chrono::duration<double, std::milli> (finish_g - start_g).count();
		std::cout << "Sampling Possible World time = " << duration_g << " ms" << std::endl;


		for (size_t i = 0; i < source_target_pairs.size(); i++) 
		{
			source_target_pair = source_target_pairs[i];
			source = source_target_pair.first;
			target = source_target_pair.second;

			// Reset var
			reliability_j.clear();
			diff_sq_sum = 0.0;
			write_flag = true;

			for (int j = 0; j < ALGO_CONF::kRepeatForVariance; j++) 
			{
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
				for (int i = 0; i < k; i++) 
				{ // kKStep for controling K sampling world
					num_reached = num_reached + traverse_run(source, target, graph_k[i]);
				}
				finish_t = std::chrono::high_resolution_clock::now();
				auto duration_t = std::chrono::duration<double, std::milli> (finish_t - start_t).count();
				std::cout << "Traversal time = " << duration_t << " ms" << std::endl;
				std::cout<< "num_reached: "<< num_reached << std::endl;

				// Calculate reliability
				reliability = num_reached / (double)k;

				// Stop time
				finish = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration<double, std::milli> (finish - start).count();

				std::cout << "Current utilize : K=" << k << " possible world." << std::endl; 
				std::cout << "Reliability Estimator, R^ (" << source << ", " << target << ") = " << reliability << std::endl;
				std::cout << "Execution time = " << duration << " ms" << std::endl << std::endl;

				if (write_flag) 
				{
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
			for (int j = 0; j < ALGO_CONF::kRepeatForVariance; j++) 
			{
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



int traverse_run(uint source, uint target, std::vector<initial_vertex> graph)
{
	std::queue<uint> worklist;
	std::set<uint> explored;
	uint v, w;

	// Add source in worklist
	worklist.push(source);
	explored.insert(source);

	if (source == target) 
	{
		return 1;
	}

	while (!worklist.empty())
	{
		v = worklist.front();
		worklist.pop();
		// T -> S: Iterate through all ingoing edges from s -> t
		uint vdegree = graph.at(v).nbrs.size();
		if ( vdegree != 0) {
			for( uint i = 0; i < vdegree; i++){
				w = graph.at(v).nbrs.at(i).tgtIndex;
				if ( w == target ) {
					return 1;
				}
				// if not explored, add to worklist
				if (explored.count(w) == 0) {
					worklist.push(w);
					explored.insert(w);
				}
			}
		}
	}
	// Target not found
	return 0;
}


}


#endif

