#ifndef MEM_ALLOC_GRAPH_CUH
#define MEM_ALLOC_GRAPH_CUH

#include "../ugraph_io/ugraph_structures.h"

namespace bpgraph
{
uint mem_alloc_graph(
    std::vector<initial_vertex>* init_graph,
    const uint num_edge,
    std::vector<std::vector<uint> > path_graph,
    uint * d_vertex_offset,
	uint * d_edge_offset,
	float * d_edge_probability,
	uint * d_edge_in_path,
	uint * d_edge_id,
    const int single_device_id
) 
{
    uint num_vertex = init_graph->size();
    uint vertex_offset[num_vertex+1] = {0};
    uint edge_offset[num_edge] = {0};
    float edge_probability[num_edge] = {0.0};
    for(int vertex_id = 0; vertex_id < num_vertex; ++vertex_id ) 
    {
        initial_vertex i_ver = init_graph->at(vertex_id);
        uint i_nbr_num = i_ver.nbrs.size();
        uint i_id_offset = vertex_offset[vertex_id];
        for(int i_edge= 0; i_edge < i_nbr_num ; i_edge ++) {
            edge_offset[i_id_offset + i_edge] = i_ver.nbrs.at(i_edge).tgtIndex;
            edge_probability[ i_id_offset + i_edge ] = i_ver.nbrs.at(i_edge).edgeValue.probability.at(0);
            //std::cout << "Edge < " << edge_offset[i_id_offset + i_edge] << "> = " 
             //   <<  edge_probability[ i_id_offset + i_edge ]<< std::endl;
        }
        vertex_offset[vertex_id + 1] = i_nbr_num + vertex_offset[vertex_id]; 
    }

    uint num_path = path_graph.size();
    uint edge_in_path[num_path+1] = {0};
    uint total_length = 0;
    for(int i_path = 0; i_path < num_path; i_path++ ) {
        edge_in_path[i_path + 1] += path_graph[i_path].size();
        total_length += path_graph[i_path].size();
    }
    uint edge_id[ total_length];
    for(int i=0; i< num_path; i++) {
        for(int j = 0; j < path_graph[i].size(); j++){
            edge_id[ i*path_graph[i].size()+j ] = path_graph[i][j];
            //std::cout << edge_id[i*path_graph[i].size()+j] << " ";
        }
        //std::cout<< std::endl;
    }

    //Data transfer
    //std::cout << "Allocat ===========: " << vertex_offset[4] << std::endl;
    cudaMemcpy(d_vertex_offset, vertex_offset , (num_vertex+1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_offset, edge_offset, num_edge * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_probability, edge_probability, num_edge * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_in_path, edge_in_path, (num_path+1) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edge_id, edge_id, num_edge * sizeof(uint), cudaMemcpyHostToDevice);
    //Init kernel parameters
   // cudaMemcpy(d_curr_q, &source, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemset(d_next_q_size, 0, sizeof(int));
    //distance_check.resize(inMemG.num_vertex_m);
    //std::fill(distance_check.begin(), distance_check.end(), -1);
    //distance_check[source] = 0;
    //cudaMemcpy(d_distance, distance_check.data(), inMemG.num_vertex * sizeof(int), cudaMemcpyHostToDevice);


}

void mem_alloc_st_pairs(
    std::vector<std::pair<uint, uint>> source_target_pairs,
    uint * d_sources,
	uint * d_targets,
    uint num_st,
    const int device_id
)
{
    uint sources[num_st], targets[num_st];
    for (size_t pair_id = 0; pair_id < source_target_pairs.size(); pair_id++) {
		std::pair<uint, uint> source_target_pair = source_target_pairs[pair_id];
		sources[pair_id] = source_target_pair.first;
		targets[pair_id] = source_target_pair.second;
    }

    cudaMalloc(&d_sources, (num_st) * sizeof(uint));
    cudaMalloc(&d_targets, (num_st) * sizeof(uint));

    cudaMemcpy(d_sources, sources , (num_st) * sizeof(uint), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, targets , (num_st) * sizeof(uint), cudaMemcpyHostToDevice);

}




}

#endif // !MEM_ALLOC_GRAPH_CUH
