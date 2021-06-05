

#ifndef PATH_SAMPLING_MONTE_CARLO_H_
#define PATH_SAMPLING_MONTE_CARLO_H_

#include <queue>
#include <set>

#include "../ugraph_io/ugraph_structures.h"
#include "../utils/memory_monitor.h"
#include "../utils/convergence_helper.h"
#include "../ugraph_io/file_io.h"
#include "../utils/globals.h"

#define DISTANCE_CONSTRAIN 10

namespace CPU_ALGOS{

std::vector< std::vector<uint> > generate_path_graph(std::vector<std::pair<uint, uint>> st_pairs, 
	std::vector<initial_vertex> & graph);
int path_monte_carlo_run(uint source, uint target, std::vector< std::vector<uint> > & graph);


//void find_k_monte_carlo(Graph & graph)
void find_k_path_single_monte_carlo(std::vector<initial_vertex> & graph, uint nEdges, std::ifstream & stList)
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

	std::cout << std::endl << "Reading Source-Target file..." << std::endl;
	std::vector<std::pair<uint, uint>> source_target_pairs(0);
	uint nQuery = read_stlist(stList, source_target_pairs);

	memory_monitor mm = memory_monitor();
	std::thread t1(&memory_monitor::update_peak_memory, std::ref(mm));
	t1.detach();
    
	
	while ( fabs(curr_avg_r - prev_avg_r) > ALGO_CONF::kReliabilityThreshold && k < ALGO_CONF::kMaximumRound) { /*k < k_limit*/
		// Step up k
		k += ALGO_CONF::kKStepUp;
		std::cout << "k = " << k << std::endl;

		// Reset var
		reliability_k.clear();

		std::vector< std::vector<uint> > path_graph = generate_path_graph(source_target_pairs, graph);

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

            #pragma omp parallel for num_threads(1)
				for (int i = 0; i < k; i++) { // kKStep for controling K sampling world
					int is_find = 1; //path_monte_carlo_run(source, target, path_graph);
					num_reached = num_reached + is_find;
					if(is_find) 
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
				auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();

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
	mm.stop_monitoring();
}

std::vector< std::vector<uint> > generate_path_graph(std::vector<std::pair<uint, uint>> st_pairs, 
	std::vector<initial_vertex> & graph)
{
	std::vector<uint> worklist;
	std::set<uint> explored;
	std::vector< std::vector<uint> > paths;
	std::vector< std::vector<uint> > path_graph(0);
	for (size_t i = 0; i < st_pairs.size(); i++) {
		std::pair<uint, uint> source_target_pair = st_pairs[i];
		uint source = source_target_pair.first;
		uint target = source_target_pair.second;
		std::cout << "-- Generating paths of ( "<< source << " -> " << target << " )" << std::endl;
		worklist.push_back(source);
		explored.insert(source);
		// BFS for leveled vertex, max level
		//initial_path cur_path;
		uint v, w;
		while (!worklist.empty())
		{
			v = worklist.back();
			uint vdegree = graph.at(v).nbrs.size();
			if ( vdegree != 0) {
				uint i = 0;
				for( i = 0; i < vdegree; i++) {
					w = graph.at(v).nbrs.at(i).tgtIndex;
					// edge ee = graph.at(v).nbrs.at(i).edgeValue;
					if(worklist.size() == DISTANCE_CONSTRAIN) {
						worklist.pop_back();
					}
					if ( w == target ) {
						worklist.push_back(w);
						path_graph.push_back(worklist);
						worklist.pop_back();
						//worklist.pop_back();
					} else if (explored.count(w) == 0) {
						worklist.push_back(w);
						explored.insert(w);
						break;
						//cur_path.align_vertex.push_back(w);
					}
				}
				if(i == vdegree) worklist.pop_back();
			} else {
				worklist.pop_back();
			}
		}
		for(int i = 0; i< path_graph.size(); i++) {
		for(int j = 0; j < path_graph[i].size(); j++)
			std::cout<< path_graph[i][j] << " ";
		std::cout<< std::endl;
	}
	}
	
	return path_graph;
}



}


#endif

