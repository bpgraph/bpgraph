#ifndef REORGANIZE_UNCERTAIN_GRAPH_CUH
#define REORGANIZE_UNCERTAIN_GRAPH_CUH

#define DIAMETER 100

#include <algorithm>
#include <queue>

#include <stack>

using namespace std;

namespace bpgraph
{
std::vector< std::vector<uint> > reorganize_uncertain_graph(
    std::vector<std::pair<uint, uint>> st_pairs, 
    std::vector<initial_vertex> & graph,
    uint edge_num)
{
	//std::vector< std::vector<uint> > paths;
	std::vector< std::vector<uint> > path_graph(0);
	for (size_t i = 0; i < st_pairs.size(); i++) {
		std::vector<neighbor> worklist;
		std::vector<uint> explored;
		std::pair<uint, uint> source_target_pair = st_pairs[i];
		uint source = source_target_pair.first;
		uint target = source_target_pair.second;
		std::cout << "-- Generating paths of ( "<< source << " -> " << target << " )" << std::endl;
		//worklist.push_back(source);
		std::vector<neighbor> s_nei = graph.at(source).nbrs;
		worklist.insert(worklist.end(), s_nei.begin(), s_nei.end());
		explored.push_back(source);
		// BFS for leveled vertex, max level
		//initial_path cur_path;
		uint v, w;
		while (!worklist.empty())
		{
			neighbor v = worklist.back();
			uint vdegree = graph.at(v.tgtIndex).nbrs.size();
			if ( vdegree != 0) {
				uint i = 0;
				for( i = 0; i < vdegree; i++) {
					w = graph.at(v.tgtIndex).nbrs.at(i).tgtIndex;
					// edge ee = graph.at(v).nbrs.at(i).edgeValue;
					if(worklist.size() > DIAMETER) {
						worklist.pop_back();
					}
					if ( w == target ) {
						//explored.push_back(w);
						path_graph.push_back(explored);
						explored.pop_back();
						worklist.pop_back();
						//worklist.pop_back();
					//} else if (explored.count(w) == 0) {
					} else {
						auto iter = std::find_if(explored.begin(), explored.end(), [&](const uint& item)->bool
				             { return (item == w); });
						if (iter != explored.end()) {
							std::vector<neighbor> s1_nei = graph.at(w).nbrs;
							worklist.insert(worklist.end(), s1_nei.begin(), s1_nei.end());
							//worklist.push_back(w);
							explored.push_back(w);
							//break;
							//cur_path.align_vertex.push_back(w);
						}
					}
				}
				//if(i == vdegree) worklist.pop_back();
			} else {
				worklist.pop_back();
				//explored.pop_back();
			}
		}
	}	
	return path_graph;
}


inline void print_path(std::vector<uint> path)
{
    std::cout<<"[ ";
    for(int i=0;i<path.size();++i)
    {
        std::cout<<path[i]<<" ";
    }
    std::cout<<"]"<<std::endl;
}


//graph[i][j] stores the j-th neighbour of the node i
void do_stack_traversal(uint start, uint end, const vector<initial_vertex > &graph, uint GLOBAL_DIST_CONSTRAIN) 
{
   //initialize:
   //remember the node (first) and the index of the next neighbour (second)
   typedef pair<uint, uint> State;
   stack<State> to_do_stack;
   vector<uint> path; //remembering the way
   vector<bool> visited(graph.size(), false); //caching visited - no need for searching in the path-vector 


   //start in start!
   to_do_stack.push(make_pair(start, 0));
   visited[start]=true;
   path.push_back(start);

   while(!to_do_stack.empty())
   {
      State &current = to_do_stack.top();//current stays on the stack for the time being...

      if (current.first == end || current.second == graph[current.first].nbrs.size() || path.size() == GLOBAL_DIST_CONSTRAIN)//goal reached or done with neighbours?
      {
          if (current.first == end)
            print_path(path);//found a way!

          //backtrack:
          visited[current.first]=false;//no longer considered visited
          path.pop_back();//go a step back
          to_do_stack.pop();//no need to explore further neighbours         
      }
      else{//normal case: explore neighbours
          uint next=graph[current.first].nbrs.at(current.second).tgtIndex;
          current.second++;//update the next neighbour in the stack!
          if(!visited[next]){
               //putting the neighbour on the todo-list
               to_do_stack.push(make_pair(next, 0));
               visited[next]=true;
               path.push_back(next);
         }      
      }
  }
}

bool isadjacency_node_not_present_in_current_path(int node, std::vector<uint>path)
{
    for(int i=0; i < path.size(); ++i)
    {
        if(path[i] == node)
        	return false;
    }
    return true;
}

std::vector< std::vector<uint> > reorganize_uncertain_graph(
    std::vector<std::pair<uint, uint>> st_pairs, 
    std::vector<initial_vertex> & graph,
    uint edge_num,
	uint GLOBAL_DIST_CONSTRAIN)
{
	//std::vector< std::vector<uint> > paths;
	std::vector< std::vector<uint> > path_graph(0);
	for (size_t i = 0; i < st_pairs.size(); i++) {
		std::pair<uint, uint> source_target_pair = st_pairs[i];
		uint source = source_target_pair.first;
		uint target = source_target_pair.second;
		std::cout << "-- Generating paths of ( "<< source << " -> " << target << " )" << std::endl;
		//do_stack_traversal(source, target, graph, GLOBAL_DIST_CONSTRAIN);
		std::vector<uint> path;
		path.push_back(source);
		std::queue<std::vector<uint> > current_path;
		current_path.push(path);

		while(!current_path.empty())
		{
			path = current_path.front();
			current_path.pop();

			int last_nodeof_path = path[path.size()-1];
			if(last_nodeof_path == target)
			{
				path_graph.push_back(path);
			}
			else
			{
				//path_graph.push_back(path);
				std::vector<neighbor> last_neighbors = graph.at(last_nodeof_path).nbrs;
				for(int i = 0;i < last_neighbors.size();++i)
				{
					uint tgtV = last_neighbors.at(i).tgtIndex;
					if(isadjacency_node_not_present_in_current_path(tgtV, path))
					{
						std::vector<uint> new_path(path.begin(),path.end());
						new_path.push_back(tgtV);
						current_path.push(new_path);
					}
				}
			}
		}
	}

	return path_graph;
}

    
}


#endif // !REORGANIZE_UNCERTAIN_GRAPH_CUH
