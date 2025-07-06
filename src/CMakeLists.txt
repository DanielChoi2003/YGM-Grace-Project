# Copyright 2019-2024 Lawrence Livermore National Security, LLC and other YGM
# Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: MIT

add_ygm_executable(grey_wolf)
add_ygm_executable(triangle_centrality)
#add_ygm_executable(bareiss)
add_ygm_executable(testing)

function(setup_ygm_target exe_name)
    if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
        target_link_libraries(${exe_name} PRIVATE rt)
    endif ()
    target_link_libraries(${exe_name} PRIVATE ygm::ygm)
    
endfunction()


function(add_gctc_executable name source)
    add_executable(${name} ${source})
    #add_common_compile_options(${name})
    setup_ygm_target(${name})
    target_include_directories(${name} PUBLIC "${CMAKE_SOURCE_DIR}/include")
endfunction()

add_gctc_executable(ygm_graph_challenge_tc ygm_graph_challenge_tc.cpp)