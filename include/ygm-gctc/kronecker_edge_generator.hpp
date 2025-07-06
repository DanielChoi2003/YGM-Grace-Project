// Copyright 2019-2021 Lawrence Livermore National Security, LLC and other YGM
// Project Developers. See the top-level COPYRIGHT file for details.
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <fstream>

#include <ygm-gctc/rmat_edge_generator.hpp>
#include <ygm/comm.hpp>
#include <ygm/detail/ygm_cereal_archive.hpp>

namespace gctc {


// namespace: to group related functions, data, variables. Acts like global / static members.
//            Avoids naming collision
namespace kronecker {


// I don't know what type T and W will be. It will be determined during compile time.
// it will also return type T, which is also determined during compile time
template <typename T, typename W>
T read_replicated_graph_file(ygm::comm &world, std::string filename,
                             std::vector<std::tuple<T, T, W>> &edge_list) {
  edge_list.clear();  // accessed with reference. thus empty it before use.
  T num_vertices(0);  // declare a new variable of type T and initialize it to 0.
  auto num_vertices_ptr = world.make_ygm_ptr(num_vertices);  // similar to static? Why is not used?
  auto edge_list_ptr = world.make_ygm_ptr(edge_list);

  if (world.rank0()) {
    std::ifstream filestream(filename);

    if (filestream.is_open()) {
      std::string line;
      if (!std::getline(filestream, line)){ // extract a line
        std::cerr << "Empty file\n";
        exit(-1);
      }
      std::istringstream iss(line);
      iss >> num_vertices;  // extract the number of vertices (on the first line)
      if (line.find(" ") != line.npos) {
        std::cout << iss.str() << std::endl;
        std::cerr << "First line of input has too many values\n";
        exit(-1);
      }
      while (std::getline(filestream, line)) {
        std::istringstream iss2(line);
        T src, dest;
        W wgt;
        if (!(iss2 >> src >> dest >> wgt)) { // extracting src, dest, and weight
          std::cerr << "Malformed line in input\n";
          exit(-1);
        } else {  // creating an undirected edge
          edge_list.push_back(std::make_tuple(src, dest, wgt));
          // Forcing to be symmetric, at least for now...
          edge_list.push_back(std::make_tuple(dest, src, wgt));
        }
      }
      filestream.close();
    } else {
      std::cerr << "Unable to open file " << filename << std::endl;
      exit(-1);
    }
  }
  ygm::bcast(num_vertices, 0, world); // broadcasting the number of vertices and edge list to all other ranks
  ygm::bcast(edge_list, 0, world);

  world.barrier();

  return num_vertices;
}

} // namespace kronecker

template <
    typename edge_data_type = uint32_t, // to differentiate?
    typename C1 = std::vector<std::tuple<uint32_t, uint32_t, edge_data_type>>,
    typename C2 = std::vector<std::tuple<uint32_t, uint32_t, edge_data_type>>> // a vector of tuple that contains 3x 32 byte ints
class kronecker_edge_generator {
public:
  using vertex_t = uint64_t;
  using edge_data_t = edge_data_type; // uint32_t
  using edge_t = std::tuple<uint64_t, uint64_t, edge_data_t>;  // 

public:
  kronecker_edge_generator(ygm::comm &c, C1 graph1, C2 graph2,
                           uint64_t num_vertices_graph1,
                           uint64_t num_vertices_graph2)
      : m_graph1(graph1), m_graph2(graph2), m_comm(c),
        m_num_vertices_graph1(num_vertices_graph1),
        m_num_vertices_graph2(num_vertices_graph2) {
    m_vertex_scale =
        (uint64_t)ceil(log2(m_num_vertices_graph1 * m_num_vertices_graph2));
  }

  kronecker_edge_generator(ygm::comm &c, std::string filename1,
                           std::string filename2)
      : m_comm(c) {

        // read_replicated_graph_file already broadcasts number of vertices and the edge list to all ranks
    m_num_vertices_graph1 =
        kronecker::read_replicated_graph_file(m_comm, filename1, m_graph1); // storing the number of vertices for graph 1
    m_num_vertices_graph2 =
        kronecker::read_replicated_graph_file(m_comm, filename2, m_graph2);

    m_vertex_scale =
        (uint64_t)ceil(log2(m_num_vertices_graph1 * m_num_vertices_graph2));
    if (m_comm.rank0()) {
      std::cout << "Vertex Scale: " << m_vertex_scale << std::endl;
    }
  }

  template <typename Function> void for_all(Function fn) {
    size_t graph1_pos = m_comm.rank();

    while (graph1_pos < m_graph1.size()) {
      const edge_t &graph1_edge = m_graph1.at(graph1_pos); // every rank starts at a different position of m_graph1

      std::for_each(m_graph2.begin(), m_graph2.end(),
                    [fn, &graph1_edge, this](const edge_t &graph2_edge) {
                      vertex_t row1 = std::get<0>(graph1_edge); // source of e1
                      vertex_t col1 = std::get<1>(graph1_edge); // dest of e1
                      edge_data_t val1 = std::get<2>(graph1_edge); // weight of e1

                      vertex_t row2 = std::get<0>(graph2_edge); // source of e2
                      vertex_t col2 = std::get<1>(graph2_edge); // dest of e2
                      edge_data_t val2 = std::get<2>(graph2_edge); // weight of e2

                      // kronecker formula
                      vertex_t row = row1 * this->m_num_vertices_graph2 + row2; // m_num_vertices_graph2: number of vertices 
                      vertex_t col = col1 * this->m_num_vertices_graph2 + col2;
                      edge_data_t val = val1 * val2;

                      // add scrambling
                      row = detail::hash_nbits(row, m_vertex_scale); // m_vertex_scale defines the range of the scrambled ID
                      col = detail::hash_nbits(col, m_vertex_scale);
                      fn(row, col, val); // calls the user function with the new value
                    });

      graph1_pos += m_comm.size();
    }
  }

  void clear() {
    m_graph1.clear();
    m_graph1.clear();
  }

private:
  ygm::comm &m_comm;

  C1 m_graph1;
  C2 m_graph2;
  uint64_t m_num_vertices_graph1;
  uint64_t m_num_vertices_graph2;
  uint64_t m_vertex_scale;
};

} // namespace gctc