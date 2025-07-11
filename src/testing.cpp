#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <cereal/types/set.hpp>


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

int main(int argc, char** argv){
    
    ygm::comm world(&argc, &argv);
    
    ygm::container::map<int, vert_info> vertex_map(world);

   

    world.cout("finished");
    return 0;
}