// Copyright 2019-2025 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#include <boost/unordered/unordered_flat_set.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <ygm-gctc/kronecker_edge_generator.hpp>
#include <ygm/comm.hpp>
#include <ygm/container/counting_set.hpp>


// finds the appropriate graph files based on the user input
std::pair<std::string, std::string> parse_cmdline(int argc, char **argv,
                                                  ygm::comm &comm) {

  YGM_ASSERT_RELEASE(argc <= 2);

  int scale = 20; 
  if (argc == 2) {
    scale = atoi(argv[1]);
  }

  if (scale < 20 || scale > 42) {
    comm.cout0("ERROR:  Minimum scale is 20, Maximum scale is 42");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int A_scale = scale / 2;
  int B_scale = scale / 2 + (scale % 2 == 1);

  std::stringstream A_sstr, B_sstr;

  
  A_sstr << "./data/A_G500_S_testing_tc.edges";
  B_sstr << "./data/B_G500_S_testing_tc.edges";

  // A_sstr << "./data/A_G500_S" << A_scale << "_tc.edges";
  // B_sstr << "./data/B_G500_S" << B_scale << "_tc.edges";

  comm.cout0("SCALE: ", scale);
  comm.cout0("Input A: ", A_sstr.str());
  comm.cout0("Input B: ", B_sstr.str());

  return {A_sstr.str(), B_sstr.str()};
}

int main(int argc, char **argv) {
  ygm::comm world(&argc, &argv);

  static ygm::comm& s_world = world;
  {

    world.cout0("YGM Graph Challenge -- Triangle Counting");
    static int s_world_size = world.size();

    auto names = parse_cmdline(argc, argv, world);
    world.welcome();

    gctc::kronecker_edge_generator kron(world, names.first, names.second); // "./data/A_G500_S" << A_scale << "_tc.edges"
    uint64_t global_num_edges = 0;
    uint64_t global_kron_triangles = 0;

    //
    // Count degrees
    ygm::container::counting_set<uint64_t> vertex_degrees(world);
    boost::unordered::unordered_flat_set<uint64_t> set_my_vertices;
    world.cf_barrier(); // what's the difference from a normal barrier()?
    double cd_time_start = MPI_Wtime();
    kron.for_all(
        // this is the fn you pass into the custom for_all()
        [&global_num_edges, &global_kron_triangles, &vertex_degrees,
         &set_my_vertices](const auto row, const auto col, const auto val) { // this uses the newly calculated value of row, col, and val
          if (row < col) { // to avoid duplicates. because graph 1 and 2 are undirected, they will produce a duplicate
            ++global_num_edges;
            global_kron_triangles += val;
            vertex_degrees.async_insert(row); // counts how many times a vertex has been inserted
            vertex_degrees.async_insert(col);
            set_my_vertices.insert(row);
            set_my_vertices.insert(col);
          }
    });
    world.barrier();
    // now all ranks have access to the map of vertex -> counts
    std::map<uint64_t, uint64_t> map_my_vertices = vertex_degrees.gather_keys(
        std::vector<uint64_t>(set_my_vertices.begin(), set_my_vertices.end()));
    world.barrier();
    vertex_degrees.clear();
    set_my_vertices.clear();
    double cd_time_end = MPI_Wtime();
    global_num_edges = ygm::sum(global_num_edges, world);
    global_kron_triangles = ygm::sum(global_kron_triangles, world) / 3;
    world.cout0("global_num_edges = ", global_num_edges);
    world.cout0("global_kron_triangles = ", global_kron_triangles);
    world.cout0("map_my_vertices.size() = ", map_my_vertices.size());
    world.cout0("Degree Count Time = ", cd_time_end - cd_time_start);

    //
    //  Build dodgr edge list
    struct dodgr_edge {
      uint32_t source;  // local row index
      uint32_t global_source; // actual source vertex ID
      uint64_t dest;
      uint32_t dest_degree;
    } __attribute__((packed)); // tells the compiler to not add byte padding between the data members
    static std::deque<dodgr_edge> dodgr_edges_1d; // degree order directed graph
    world.cf_barrier();
    double rh_time_start = MPI_Wtime();
    kron.for_all([&map_my_vertices, &world](const auto row, const auto col,
                                            const auto val) {
      if (row < col) { 
        YGM_ASSERT_RELEASE(map_my_vertices.count(row) > 0);
        YGM_ASSERT_RELEASE(map_my_vertices.count(col) > 0);
        uint64_t row_degree = map_my_vertices[row];
        uint64_t col_degree = map_my_vertices[col];
        uint64_t source(0);
        uint64_t target(0);
        uint32_t target_degree(0);
        if (row_degree > col_degree) { // the edge redirected towards the higher degree vertex
          source = col;
          target = row;
          target_degree = row_degree;
        } else if (row_degree < col_degree) {
          source = row;
          target = col;
          target_degree = col_degree;
        } else { // equal degree. tie breaker
          if (row > col) {
            source = col;
            target = row;
            target_degree = row_degree;
          } else if (row < col) {
            source = row;
            target = col;
            target_degree = col_degree;
          } else {
            std::cout << "Found a self edge!?!?!" << std::endl;
          }
        }

        // to a rank (source % world.size())
        world.async(
            source % world.size(),
            [](uint64_t source, uint64_t target, uint32_t target_degree) {
              dodgr_edges_1d.push_back({
                  uint32_t(source / s_world_size), // what is the reason?
                  target,
                  target_degree
              });
              s_world.cout(source, "->", target);
            },
            source, target, target_degree);
      }
    });
    world.barrier();
    kron.clear();
    double rh_time_end = MPI_Wtime();
    map_my_vertices.clear();
    std::sort(dodgr_edges_1d.begin(), dodgr_edges_1d.end(),
              [](const auto &ta, const auto &tb) {  // if source is less than, put it first. If equal, compare the dest.
                return std::make_pair(ta.source, ta.dest) <
                       std::make_pair(tb.source, tb.dest);
              });
    static std::vector<size_t> row_jump_index;
    int64_t cur_row = 0;
    row_jump_index.push_back(0);
    for (size_t i = 0; i < dodgr_edges_1d.size(); ++i) {
      size_t local_row_index = dodgr_edges_1d[i].source;
      while (local_row_index > cur_row) { // For any vertex u, where do its edges begin in the edge list?
        ++cur_row;
        row_jump_index.push_back(i);
        YGM_ASSERT_RELEASE(cur_row == row_jump_index.size() - 1);
      }
    }
    static size_t local_largest_vertex = dodgr_edges_1d.back().source;
    row_jump_index.push_back(dodgr_edges_1d.size()); // padding to make accessing safe
    row_jump_index.push_back(dodgr_edges_1d.size());
    row_jump_index.push_back(dodgr_edges_1d.size());
    row_jump_index.push_back(dodgr_edges_1d.size());
    YGM_ASSERT_RELEASE(local_largest_vertex < row_jump_index.size());
    static std::vector<size_t> dodgr_edges_1d_targets(dodgr_edges_1d.size());
    for (size_t i = 0; i < dodgr_edges_1d.size(); ++i) {
      dodgr_edges_1d_targets[i] = dodgr_edges_1d[i].dest;
    }
    world.barrier();
    world.cout0("Build dodgr time = ", rh_time_end - rh_time_start);
    world.cout0("Global max local vertex set size = ",
                ygm::max(row_jump_index.size(), world));

    //
    // Generate wedges and check!!
    world.cf_barrier();
    world.stats_reset();
    double wc_time_start = MPI_Wtime();
    size_t global_wedge_checks(0);
    static size_t global_triangles_found(0);
    /*
      original vertex node number: source
      owner of the vertex: source % world.size
      dodgr_edges_1d[i].source = local edge list's index

      so "source" here is simply the index
      "dest" here is the actual destination vertex number
    */
    for (size_t i = 0; i < dodgr_edges_1d.size(); ++i) {
      uint64_t t_i = dodgr_edges_1d[i].source; // local?
      uint64_t t_j = dodgr_edges_1d[i].dest;
      uint64_t t_j_deg = dodgr_edges_1d[i].dest_degree;
      for (size_t j = i + 1; j < dodgr_edges_1d.size(); ++j) {
        if (t_i == dodgr_edges_1d[j].source) { // this is comparing the local edge list index, but if index is equal == the source vertex is equal
          uint64_t t_k = dodgr_edges_1d[j].dest;
          uint64_t t_k_deg = dodgr_edges_1d[j].dest_degree;
          global_wedge_checks++;

          int query_dest_rank = 0;

          struct packed {
            uint32_t query_source;
            uint64_t query_target;
          } __attribute__((packed));

          packed p;

          if (t_j_deg < t_k_deg) {  // mark the higher degree vertex as the destination
            p.query_source = t_j / s_world_size;
            p.query_target = t_k;
            query_dest_rank = t_j % s_world_size;
          } else if (t_j_deg > t_k_deg) {
            p.query_source = t_k / s_world_size;
            p.query_target = t_j;
            query_dest_rank = t_k % s_world_size;
          } else if (t_j < t_k) { // if the degree is equal, break the tie with vertex number
            p.query_source = t_j / s_world_size;
            p.query_target = t_k;
            query_dest_rank = t_j % s_world_size;
          } else {
            p.query_source = t_k / s_world_size;
            p.query_target = t_j;
            query_dest_rank = t_k % s_world_size;
          }

          /*  MY EDIT
            Need to increment the other two vertices' local triangle count

            The problem arises when we have to get the sum of total triangle count because originally, it was 
            implemented with an undirected graph. but need to figure a way with the degree order directed graph.
          
            How to get the core count of a vertex? 
            Adding triangle count as triangle is found leads to overcounting. If a neighboring vertex participates in
            two triangles, it will count double

            First, it needs to gather triangle counts for all vertices.
            Second, gather all triangle neighbors.
              Call the neighbors and make the neighbors add to the calling vertex's core count
          */
          world.async(query_dest_rank, [p, t_i, t_j, t_k]() { // difference from async_visit?
            size_t local_row_index = p.query_source;
            if (local_row_index <= local_largest_vertex) {
              // lower_bound finds the first occurrence of val or a value that is greater if the exact "val" is not found
              auto itr =
                  std::lower_bound(dodgr_edges_1d_targets.begin() +
                                       row_jump_index[local_row_index],
                                   dodgr_edges_1d_targets.begin() +
                                       row_jump_index[local_row_index + 1],
                                   p.query_target);
              // if it found p.query_target and it matches the destination vertex -> triangle found
              if (itr != dodgr_edges_1d_targets.begin() +
                             row_jump_index[local_row_index + 1] &&
                  *itr == p.query_target) {
                global_triangles_found++; // found a triangle!

                s_world.cout("triangle found: ", t_i, " -> ", t_j, " -> ", t_k);

                /*
                  increment the local triangle count for:
                  t_i, t_j, t_k

                */

              }
            }
          });
        } else {
          break; // end of t_i's edges
        }
      }
    }
    world.barrier();
    world.stats_print("TC_TIME");
    double wc_time_end = MPI_Wtime();

    world.barrier();
    global_wedge_checks = ygm::sum(global_wedge_checks, world);
    global_triangles_found = ygm::sum(global_triangles_found, world);
    world.cout0("global_wedge_checks = ", global_wedge_checks);
    world.cout0("global_triangles_found = ", global_triangles_found);
    world.cout0("global_kron_triangles = ", global_kron_triangles);
    world.cout0("TC Time = ", wc_time_end - wc_time_start);
    if (global_kron_triangles == global_triangles_found) {
      world.cout0("!!! PASSED !!!");
    } else {
      world.cout0(
          "*** FAIL *** FAIL *** FAIL *** FAIL *** FAIL *** FAIL *** FAIL *** "
          "FAIL *** FAIL ***");
    }
    world.barrier();


    //
    // get the core, non-core, and total triangle count
  }
  return 0;
}

/*
  Questions:
  1. How to confirm correctness?
  2. How to ensure the completion of nested async?
  3. How to get the original source back? To get the triangle centrality, I need the original source vertex.
    Add the global source ID into dodgr?
*/
