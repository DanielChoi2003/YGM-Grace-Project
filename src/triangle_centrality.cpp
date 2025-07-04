#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/set.hpp>
#include <ygm/container/bag.hpp>
#include <ygm/container/array.hpp>
#include <vector>
#include <utility> // std::pair
#include <queue> // std::priority_queue
#include <functional> // std::greater
#include <filesystem>
#include <ygm/io/csv_parser.hpp>
#include <algorithm>
#include <random>


// implementation of triangle counting with 2-core decomposition And Degree Ordered Directed graph

struct vert_info {

  template <class Archive>
  // what does this do?
  // i assume this determines what data is package when it is sent to another rank
  void serialize( Archive & ar )
  {
    ar(low_adj, high_adj);
  }

  std::vector<int> adj;
  std::vector<int> low_adj;
  std::vector<int> high_adj;

  //int degree = 0;
  int marked = 0;
  int triangle_count = 0;
  std::vector<int> triangle_neighbors;

  int core_count = 0;
  int non_core_count = 0;
  // sum of non-core and core (from all neighbors)
  int total_count = 0;

  double triangle_centrality = 0;

};

//

using graph_type = ygm::container::map<int, vert_info >;

void add_edge(graph_type& graph,
              int src, int dest) {

  auto inserter = [](int src, vert_info& vi, int dest) {

    if(dest > src){
      vi.high_adj.push_back(dest);
    }
    else{
      vi.low_adj.push_back(dest);
    }

    vi.adj.push_back(dest);

    // vi.degree = vi.adj.size();
  };

  graph.async_visit(src, inserter, dest);
  graph.async_visit(dest, inserter, src);
}

int main(int argc, char** argv) {
 
  ygm::comm world(&argc, &argv);
  world.welcome();

  graph_type graph(world);

  static graph_type& s_graph = graph;
  static ygm::comm& s_world = world;
  static int local_triangle_number = 0;
  static int global_triangle_number = 0;

  static double local_max_triangle_centrality = -1;
  static double global_max_triangle_centrality = -1;


  auto aggregator = [](int a, int b){
    return a + b;
  };



  // STEP 1: insert nodes into the undirected graph

  #define CSV_READER

  #ifdef CSV_READER

  std::vector<std::string> filenames = {"../data/zachary_karate.csv"};

  

  ygm::io::csv_parser parser(world, filenames);
  parser.for_all([](ygm::io::detail::csv_line line){
    std::string line1 = line[1].as_string();
    line1.erase(0, line1.find_first_not_of(" \t\r\n"));  // trim leading whitespace
    line1.erase(line1.find_last_not_of(" \t\r\n") + 1);

    long long vertex_a = line[0].as_integer();
    long long vertex_b = std::stoi(line1);
    long long vertex_one = std::min(vertex_a, vertex_b);
    long long vertex_two = std::max(vertex_a, vertex_b);
    //std::cout << "vertex 1: " << vertex_one << std::endl;
    //std::cout << "vertex 2: " << vertex_two << std::endl;

    // need to exclude duplicate from being added since 
    // this graph is undirected
    add_edge(s_graph, vertex_one, vertex_two);
  });
  

  world.barrier();

  #endif

  //#define random_graph

  #ifdef random_graph

  const int total_edges = 1600000;
  const int total_vertices = 100000;

  std::random_device rd;  // a seed source for the random number engine
  std::mt19937 rng(world.rank());
  std::uniform_int_distribution<int> dist(0, total_vertices - 1); // generates a number between 0 to total_vertices - 1 (inclusive)

  static int local_edge_count = total_edges / world.size();
  int leftover = total_edges % world.size();
  local_edge_count += (world.rank() < leftover) ? 1 : 0;

  for(int i = 0; i < local_edge_count; i++){
    int u = dist(rng) % total_vertices;
    int v = dist(rng) % total_vertices;
    if(u == v){
      continue;
    }

    std::pair<int, int> e = std::minmax(u, v);

    add_edge(s_graph, e.first, e.second); 
  }

  // somehow all ranks are pre-maturely exiting the while loop. How??
  // global_edge_set.size() is giving some uninitialized / junk value.
  // workaround: each rank has its own local counter and aggregate those instead of 
  // global_edge_set.size() since the size() does not seem like a reliable way to count the number of 
  // unique edges


  /*
    NOTE: 
      global_set_edge.size() gets the size across all ranks and performs a reduce behind the scene (?)
      Remember that calling .size() on ygm::set could lead to DEADLOCK since it requires all ranks to participate
    Evidence: using "if(world.rank0())" leads to a deadlock since rank 0 is waiting for other ranks to 
              participate but due to rank0() function, other ranks cannot, so rank 0 ends up waiting indefinitely
  */

  world.barrier();
  #endif

  static int local_count = 0;

  graph.for_all([](int src, vert_info& vi){
    local_count += vi.low_adj.size() + vi.high_adj.size();

  });

  int global_count = ygm::all_reduce(local_count, aggregator, world);

  world.cout0("total_edges: ", global_count / 2); // divide it by two, if its undirected

  world.barrier();


  /* STEP 2: 2-Core Decomposition; Remove nodes that are not part of triangles
           This was disabled for triangle centrality because the original implementation did
           not utilize this, but also we want all nodes to have a value of TC.
*/


  // static bool no_local_marked = false;
  // static bool global_decomp = false;

  // global_decomp = world.all_reduce(no_local_marked, [](bool a, bool b){
  //       return a && b;
  //   });

  // while(!global_decomp){

  //   no_local_marked = true; // assume that there is no marked vertex yet
  //   graph.for_all([](int source, vert_info& vi){
    
  //       // 1. each process computes its local vertices' degree
  //       // 2. each process then identifies vertices with degree less than 2
  //       // 3. remove the marked vertices and other processes that own the removed
  //       //    vertices will have to update the degree
  //       // 4. Repeat until all processes don't have any marked vertices

  //       if(vi.adj.size() < 2){

  //           // how to find which process owns node that are connected to this soon-to-be deleted node?

  //           // 1. go through the deleted node's adjacency list -> async_visit(every node in the list, deleted node as a parameter)
  //           // 2. the process that owns that node will go through its adjacency list and remove the deleted node

  //           // it may send a request to a node that had already been deleted -> segmentation fault
  //           // it may create a new key-value pair
  //           for(auto neighbor : vi.adj){

  //               auto remover = [](int source2, vert_info& vi2, int source){
  //                   //s_world.cerr("Running remover on ", source2, " for neighbor ", source);

  //                   auto it = std::find(vi2.adj.begin(), vi2.adj.end(), source);
  //                   if (it != vi2.adj.end()) {
  //                       vi2.adj.erase(it);
  //                   }
  //                   //s_world.cout("Erased node ", source, " from Node ", source2, "'s adjacency list");
  //               };

  //               s_graph.async_visit(neighbor, remover, source);
  //           }

  //           no_local_marked = false;

  //           // difference between erase and async_erase?
  //           s_graph.async_erase(source);
  //       }


  //   });

  //   global_decomp = world.all_reduce(no_local_marked, [](bool a, bool b){
  //       return a && b;
  //   });
  // }

  double start = MPI_Wtime();



  /*
    STEP 3: direct the edge from low degree vertex to high degree vertex (convert undirected graph into 
    degree-ordered graph)
    since this is a simple implemenation, for equal degree vertex, break the tie with higher vertex number

    This can be used to speed the processing time (not included in the runtime)
  */
   

  // graph.for_all([](int source, vert_info& vi){
  //   // for(auto kv : graph)
    
  //   // iterate through the node's adjacency list and compare its degree to
  //   // degrees of node in the list. 
  //   // 1. if its degree is larger, keep the edge
  //   // 2. if its degree is smaller, remove it
  //   // 3. if its degree is equal, remove the edge of the smaller vertex number

  //   for(auto neighbor : vi.adj){
        
  //       auto edgeRemover = [](int source2, vert_info& vi2, int degree, int source){
  //           if(vi2.degree < degree || (vi2.degree == degree && source2 < source)){
  //               auto it = std::find(vi2.adj.begin(), vi2.adj.end(), source);
  //               if (it != vi2.adj.end()) {
  //                   vi2.adj.erase(it);
  //               }
  //           }
  //           else if(vi2.degree > degree || (vi2.degree == degree && source < source2)){

  //               auto sourceRemove = [](int source, vert_info& vi, int nodeToRemove){
  //                   auto it = std::find(vi.adj.begin(), vi.adj.end(), nodeToRemove);
  //                   if (it != vi.adj.end()) {
  //                       vi.adj.erase(it);
  //                   }

  //               };

  //               s_graph.async_visit(source, sourceRemove, source2);
  //           }

  //       };

  //       s_graph.async_visit(neighbor, edgeRemover, vi.degree, source);
  //   }
  // });
   
  /*
    STEP 4: each vertex gets a list of triangle neighbors 
            (neighbors that make up triangles with the target vertex)
  */


  graph.for_all([](int v_node, vert_info& vi){
   
    // mark the neighbor with higher ID
    for(auto u_node : vi.high_adj){

      auto marker = [](int u_node, vert_info& vi){
        vi.marked++;
      };

      s_graph.async_visit(u_node, marker);
    }
  });

  world.barrier();

  graph.for_all([](int v_node, vert_info& vi){
   
    // for every neighbor with lower ID than v_node
    for(auto u_node : vi.low_adj){

      auto outer_checker = [=](int u_node, vert_info& vi){
        for(auto w_node : vi.high_adj){
          
          // if the w_node is marked, that is one triangle
          auto inner_checker = [=](int w_node, vert_info& vi){
            
            if(vi.marked > 0){
              // increment the triangle for all u, v, w, and local_total_triangle

              local_triangle_number++;

              // add the neighbors to triangle neighbor vector
              auto add_neighbor = [=](int vert, vert_info& vi, int v1, int v2){

                vi.triangle_count++;

                vi.triangle_neighbors.push_back(v1);
                vi.triangle_neighbors.push_back(v2);
              };

              s_graph.async_visit(v_node, add_neighbor, u_node, w_node);
              s_graph.async_visit(u_node, add_neighbor, v_node, w_node);
              s_graph.async_visit(w_node, add_neighbor, v_node, u_node);

            }
          };
        }
      };
    }

    // unmark them
    for(auto u_node : vi.high_adj){
      auto unmarker = [](int u_node, vert_info& vi){
        vi.marked--;
      };

      s_graph.async_visit(u_node, unmarker);
    }
  });

  world.barrier();

  global_triangle_number = ygm::all_reduce(local_triangle_number, aggregator, world);

  // for every vertex v, get a list of triangle neighbors (implement in the code above)
  // and using that list here (after all nodes got their vertex triangle count),
  // sum the core triangle count

  // to calculate all neighbors' triangle count, it requires the graph to be undirected

  // Core-count = summation of local triangle count of vertex u, where u is a triangle neighbor
  // of vertex v (AND ITSELF TOO)


  //     vi.non_core_count = vi.total_count - vi.core_count + vi.triangle_count;
  // before: 0: Vertex 1 has total count of 67, core count: 64, non-core count: 21 , and triangle count of 18
  // after: 0: Vertex 1 has total count of 67, core count: 82, non-core count: 3 , and triangle count of 18

  graph.for_all([](int vert, vert_info& vi){
    
    vi.core_count += vi.triangle_count;
    for(int neighbor : vi.triangle_neighbors){
        // s_world.cout("Vertex ", vert, " has a neighbor ", neighbor);
        // each neighbor tells the vertex v what number to add
        auto instructVert = [](int neighbor, vert_info& neighbor_vi, int target_v){
            auto addCore = [](int target_v, vert_info& target_vi, int triangle_count){
                //s_world.cout("adding core count of ", triangle_count, " to Vertex ", target_v);
                target_vi.core_count += triangle_count;
            };
            s_graph.async_visit(target_v, addCore, neighbor_vi.triangle_count);
        };
        
        s_graph.async_visit(neighbor, instructVert, vert);
    }
  });

  world.barrier();


  graph.for_all([](int vert, vert_info& vi){
     // calculate total triangle count here
    for(int neighbor_any : vi.adj){
      
      // add triangle count from neighbors even if they are not in the same triangle as the vertex v
      auto instructVert = [](int neighbor, vert_info& neighbor_vi, int target_v){
            auto addCore = [](int target_v, vert_info& target_vi, int triangle_count){
                target_vi.total_count += triangle_count;
            };
            s_graph.async_visit(target_v, addCore, neighbor_vi.triangle_count);
        };
        
        s_graph.async_visit(neighbor_any, instructVert, vert);
    }
  });

  world.barrier();

  graph.for_all([](int vert, vert_info& vi){
     // calculate the non-core count by subtracting core count from total count
    vi.non_core_count = vi.total_count - vi.core_count + vi.triangle_count;
    vi.triangle_centrality =   ((1.0 / 3) * vi.core_count + vi.non_core_count) / global_triangle_number;

    if(vi.triangle_centrality > local_max_triangle_centrality){
      local_max_triangle_centrality = vi.triangle_centrality;
    }
    
    // s_world.cout("Vertex ", vert, " has total count of ", vi.total_count, ", core count: ", vi.core_count, ", non-core count: ", vi.non_core_count, 
    //                 " , and triangle count of ", vi.triangle_count);
  });

  s_world.barrier();


  // calculate the core (neighbors who compose the triangle with you) count and 
  // non-core (neighbors who are not in your triangle) count.

  // compute the triangle centrality



  // if there is no constraint, then the same triangle can be counted 6 times.
  world.cout0("Total number of triangles: ", global_triangle_number);

  world.barrier();

  // graph.for_all([](int src, vert_info& vi){
  //   s_world.cout(src, " has triangle centrality of ", vi.triangle_centrality);
  // });

  world.barrier();

  global_max_triangle_centrality = ygm::max(local_max_triangle_centrality, world);

  world.cout0("max triangle centrality: ", global_max_triangle_centrality);

  world.barrier();


  double end = MPI_Wtime();

  if (world.rank0()) {
    std::cout << "Elapsed time: " << (end - start) << " seconds\n";
  }



  // need to count number of triangles per vertex

  return 0;
}


/*
  TriangleNeighbor Algorithm:

  graph.for_all([](int vert, vert_info& vi){


  })


*/


/*
  TC Algorithm 2:

  Call TriangleNeighbor

  for vertex "v" in V (graph):
    for triangle neighbor "u" in L(v):
      X(v) += triangle count of u
    x = X(v) + triangle count of v  // triangle count from neighbor and itself

    for neighbor "u" (any type) in N(v):
      s += triangle count of "u"
    
    y (non-core sum) = s - x + triangle count of itself

    TC(v) = (1/3 * x + y) / total num of triangle

*/

