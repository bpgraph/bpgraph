
#include "../core/api.cuh"

namespace st_query
{
    struct STQUERY : bpgraph::api::AlgoBase
    {
        STQUERY(uint source, uint target)
        {

        }

        __forceinline__ __host__ __device__ 
        virtual void initialize_empty_path(
            volatile Path* local_p,	
            Path* p	)  const override 
            {
                p.distance = 0;
            }

        __forceinline__ __host__ __device__ 
        virtual void expand_path(
            volatile Path* local_p,	
            Edge* e	)  const override 
            {
                p.distance += e.distance;
                p.probability = p.probability & e.probability;
            }

        
        __forceinline__ __host__ __device__ 
        virtual void reduce_vertex(
            Path path_array[], 
            Vertex v) const override 
            {
                for(auto p : path_array)
                {
                    if( p.distance < GLOBAL.dist_constraint 
                        && p.probability > GLOBAL.prob_constraint)
                    {
                        dist = p.distance;
                        v.probability[dist] += p.probability
                    }
                    
                }
                    
            }   
    
    }

    bool st_query( ){
        std::cout << " ST QUERY: " << std::endl;
    }
}
