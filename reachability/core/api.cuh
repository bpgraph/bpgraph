#ifndef API_CUH
#define API_CUH

#include "../ugraph_io/ugraph_structures.h"


namespace bpgraph
{
    namespace api
    {
        //template<typename VertexValue, typename EdgeWeight>
        struct AlgoBase
        {

            __forceinline__ __host__ __device__ 
            virtual void initialize_empty_path(
		            volatile initial_path* local_p,	// Address of the corresponding path in shared memory.
		            initial_path* p	)  const = 0;

            __forceinline__ __host__ __device__ 
            virtual void expand_path(
		            volatile initial_path* local_p,	// Address of the corresponding path in shared memory.
		            edge* e	)  const = 0;

            
            __forceinline__ __host__ __device__ 
            virtual void reduce_vertex(
                initial_path p[], 
                initial_vertex v) const = 0;

           
        };
    }
}

#endif // !API_CUH
