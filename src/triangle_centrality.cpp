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
    ar(dist, adj, degree);
  }

  int dist = std::numeric_limits<int>::max();
  std::set<int> unedited_adj;
  std::set<int> adj;

  int degree = 0;
  int triangle_count = 0;
  std::set<int> triangle_neighbors;
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
    vi.unedited_adj.insert(dest);
    vi.adj.insert(dest);
    vi.degree = vi.adj.size();
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

  // #define CSV_READER

  #ifdef CSV_READER

  std::vector<std::string> filenames = {"../terrorist_edges.csv"};

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

  #define random_graph

  #ifdef random_graph

  const int total_edges = 6400000;
  const int total_vertices = 400000;

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
    local_count += vi.adj.size();
  });

  int global_count = ygm::all_reduce(local_count, aggregator, world);

  world.cout0("total_edges: ", global_count / 2);



  double start = MPI_Wtime();


  /*
    STEP 3: direct the edge from low degree vertex to high degree vertex (convert undirected graph into 
    degree-ordered graph)
    since this is a simple implemenation, for equal degree vertex, break the tie with higher vertex number
  */
   

  graph.for_all([](int source, vert_info& vi){
    // for(auto kv : graph)
    
    // iterate through the node's adjacency list and compare its degree to
    // degrees of node in the list. 
    // 1. if its degree is larger, keep the edge
    // 2. if its degree is smaller, remove it
    // 3. if its degree is equal, remove the edge of the smaller vertex number

    for(auto neighbor : vi.adj){
        
        auto edgeRemover = [](int source2, vert_info& vi2, int degree, int source){
            if(vi2.degree < degree || (vi2.degree == degree && source2 < source)){
                auto it = std::find(vi2.adj.begin(), vi2.adj.end(), source);
                if (it != vi2.adj.end()) {
                    vi2.adj.erase(it);
                }
            }
            else if(vi2.degree > degree || (vi2.degree == degree && source < source2)){

                auto sourceRemove = [](int source, vert_info& vi, int nodeToRemove){
                    auto it = std::find(vi.adj.begin(), vi.adj.end(), nodeToRemove);
                    if (it != vi.adj.end()) {
                        vi.adj.erase(it);
                    }

                };

                s_graph.async_visit(source, sourceRemove, source2);
            }

        };

        s_graph.async_visit(neighbor, edgeRemover, vi.degree, source);
    }
  });

  world.barrier();

 
  
  /*
    STEP 4: each vertex gets a list of triangle neighbors 
            (neighbors that make up triangles with the target vertex)
  */


  graph.for_all([](int v_node, vert_info& vi){
   
      // QUESTION: capturing vert in checker led to segmentation fault?
    for(auto u_node : vi.adj){
      //s_world.cerr("Visiting ", u_node, " from ", vert);
      auto checker = [](const int u_node, vert_info& vi2, int v_node, std::set<int> adj){
        for(auto w_node : vi2.adj){
          // does w_node exist in the adjacency list of the source node?
          if(std::find(adj.begin(), adj.end(), w_node) != adj.end()){
            //s_world.cout("Triangle found: ", v_node, " -> ", u_node, " -> ", w_node);
            local_triangle_number++;
            // incrementing node_1's vertex triangle count
            vi2.triangle_count++;
            vi2.triangle_neighbors.insert(v_node);
            vi2.triangle_neighbors.insert(w_node);


            // make a function that increments source and w_node's triangle_count
            // and adds neighbors to the list
            auto updater = [](int target, vert_info& vi3, int neighbor1, int neighbor2){
                vi3.triangle_count++;
                vi3.triangle_neighbors.insert(neighbor1);
                vi3.triangle_neighbors.insert(neighbor2);
            };
            s_graph.async_visit(v_node, updater, u_node, w_node);
            s_graph.async_visit(w_node, updater, v_node, u_node);

            // need to increment the triangle_count for 3 nodes
          }
        }
      };

      s_graph.async_visit(u_node, checker, v_node, vi.adj);
    }

  });

  world.barrier();

  global_triangle_number = ygm::all_reduce(local_triangle_number, aggregator, world);

  world.barrier();


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
    for(int neighbor_any : vi.unedited_adj){
      
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


// Questions:
// 1. capturing integer in lambda is causing segmentation fault
//      A:  Not allowed in this version; it needs to be updated with the most recent version
// 2. how to ensure sequential printing
//      A: it is not possible since print is asynchronous (you cannot guarantee any kind of ordering)
// 3. difference between erase and async_erase -> erase does not exist for ygm::containers
// 4. does async_visit create a new key if it did not exist in ygm::map?
//      A: async_visit_if_contains
// 5. barrier does not influence / determine the behavior of async operations?
//      A: it cannot 

// 1. update to v0.8
// 2. pull from the most recent YGM
// 3. use the YGM/project-template repo
//

