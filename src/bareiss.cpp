#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <sstream>











int main(int argc, char** argv){

    ygm::comm world(&argc, &argv);

    static ygm::comm& s_world = world;

    ygm::container::map<int, std::vector<int>> matrix(world);

    static int n = 30;

    /*
        NOTE: since all ranks needs access to the first row,
              I decided to let all ranks own that row. It seemed
              inefficient for all ranks to constantly communicate
              to the rank that owns it.
    */
    

    // add the rows
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_int_distribution<> random_n(1, 10);

    if(world.rank0()){
        for(int i = 0; i < n; i++){
            std::vector<int> row;
            for(int j = 0; j < n; j++){
                row.push_back(random_n(gen));
            }
            matrix.async_insert(i, row);
            std::ostringstream oss;
            for(int num : row){
                oss << num << ", ";
            }
            std::cout << oss.str() << std::endl;
        }
        // std::vector<int> v1 = {2, 3, 1};
        // std::vector<int> v2 = {3, 7, 4};
        // std::vector<int> v3 = {1, 4, 6};
        // matrix.async_insert(0, v1);
        // matrix.async_insert(1, v2);
        // matrix.async_insert(2, v3);
    }

    world.barrier();


    double start = MPI_Wtime();

    static std::vector<std::pair<int, std::vector<int>>> gathered_rows;
    static bool flipSign = false;

    // ISSUE: there could be principal leading minor of value zero
            //        the question is what happens to the ranks that own specific row
            //        and have their rows removed and added back?
            if(divisor == 0){
                // swap with another row if the divisor is zero
                world.cout("uh oh");
                for(int i = k; i < n; i++){
                    world.cout("Row ", k, " has a zero divisor. Trying to find a non-zero divisor. checking row: ", i);
                    if(gathered_rows.at(i).second.at(k - 1) != 0){
                        divisor = gathered_rows.at(i).second.at(k - 1);
                        world.cout("found a non-zero divisor: ", divisor);


                        if(!flipSign){
                            flipSign = true;
                        }
                        else{
                            flipSign = false;
                        }
                        // get copies of the two keys and rows, remove them from the map, swap the keys and insert them back

                        if(s_world.rank0()){
                            int zero_key = gathered_rows.at(k - 1).first;
                            std::vector<int> zero_vector = gathered_rows.at(k - 1).second;

                            int non_zero_key = gathered_rows.at(i).first;
                            std::vector<int> non_zero_vector = gathered_rows.at(i).second;

                            matrix.async_erase(zero_key);
                            matrix.async_erase(non_zero_key);

                            matrix.async_insert(zero_key, non_zero_vector);
                            matrix.async_insert(non_zero_key, zero_vector);
                        }
                        
                        break;
                    }
                }
            }

    for(static int k = 0; k < n - 1; k++){ // k < n - 1

        matrix.gather(gathered_rows);
        std::sort(gathered_rows.begin(), gathered_rows.end());
        static int divisor;

        if(k == 0){
            divisor = 1;
        }
        else{
            // M[k - 1, k - 1]
            divisor = gathered_rows.at(k - 1).second.at(k - 1);
        }

        if(k != n - 2){ // the pivot in this row is the determinant and it can be zero. its fine.
            

        }

        // since the rank that owns row k has its local_row filled, empty rows will get filtered out
        // ISSUE: local_row is not populated by the time this run since it is asynchronous 
        // shared_row = ygm::all_reduce(local_row, [](std::vector<int> a, std::vector<int> b) {
        //     return a.empty() ? b : a; 
        // }, world);
        
        world.barrier();


    

        matrix.for_all([&](int row_num, std::vector<int>& row){
            if(row_num >= k + 1){
                for(int j = k + 1; j < n; j++){   
                    int prev_num = row.at(j);
                    // std::cout << "current iteration k: " << k << std::endl;
                    // std::cout << "("
                    //         << row.at(j) << " * " << gathered_rows.at(k).second.at(k)
                    //         << " - "
                    //         << row.at(k) << " * " << gathered_rows.at(k).second.at(j)
                    //         << ") / " << divisor
                    //         << std::endl;

                    
                    row.at(j) = (row.at(j) * gathered_rows.at(k).second.at(k) - row.at(k) * gathered_rows.at(k).second.at(j)) / divisor;
                    // std::cout << prev_num << " -> " << row.at(j) << " at position " << row_num << ", " << j << "." << std::endl;
                }

                row.at(k) = 0;
            }
        });


  

        gathered_rows.clear();

        world.barrier();
    }

    matrix.async_visit(n - 1, [](int row_num, std::vector<int>& row){
        if(flipSign){
            row.at(row.size() - 1) = -1 * row.at(row.size() - 1);
        }
    });

    double end = MPI_Wtime();

    if (world.rank0()) {
        std::cout << "Elapsed time: " << (end - start) << " seconds\n";
    }

    gathered_rows.clear();
    matrix.gather(gathered_rows);
    std::sort(gathered_rows.begin(), gathered_rows.end());

    if(world.rank0()){
        std::cout << "-------------------" << std::endl;
        for(std::pair<int, std::vector<int>> key_value : gathered_rows){
            std::ostringstream oss;
            for(int num : key_value.second){
                oss << num << ", ";
            }
            std::cout << oss.str() << std::endl;
        }
    }



    // if(world.rank0()){
    //     matrix.async_visit(n - 1, [](int row_num, std::vector<int>& row){
    //         s_world.cout(row.back());
    //     });
    // }
    
   

    return 0;

    /*
        Questions:
            1. what happens when a row is deleted and then added back (the same row, but with different key)
                during map.for_all()? Does the rank still have access to the row? Does it skip?
    */

    /*
        Bareiss algorithm:

        input: 2D Matrix
        output: determinant, the bottom right element of the matrix


        Pseudocode:

        M(0,0) = 1

        for k = 1 to n - 1:
            for i = k + 1 to n:
                for j = k + 1 to n:
                    M(i, j) = [M(i, j) * M(k, k) - M(i, k) * M(k, j)] / M(k-1, k-1)

        Parallelization:
            Each rank owns rows.

            parallelizing the i-loop
            "i" determines what row you are editing
            "j" determines what column you are editing

            j-loop is fully independent
            i-loop is fully indepedent (it seems so far)

            Trying to parallelize the j-loop seems inefficient (and also impossible) because
            each rank performing the calculation would have to access the same row that is owned 
            by a single rank, and that rank will be overburdened by all other ranks (n - 1) trying
            to communicate with it.
            Also, YGM performs parallelization by first dividing the data among ranks, and needs to
            run for_all(), but then how would other ranks access that specific row when they do not own it?
            Instead, they will run their own rows.




        NOTE: 
            in case M(k-1, k-1) is zero (cannot be zero since the formula uses it as a denominator),
            swap the row with a non-zero leading pricipal minor. And remember that the sign has to be swapped.
            So it's probably good to make a global variable that indicates whether the answer should be positive 
            or negative. The rank that owns the last row needs to apply the sign change (or broadcast it to rank zero or all)

            THE ISSUE IS PIVOT DEPENDENCY

        Problem A1: 
            The issue is that when a row has a future M[k-1, k-1] that is zero, then it needs to swap with a row below it
            [i, k-1] (i = k, ... , n).
            But since my current implementation parallelizes by row (multiple ranks calculate multiple rows
            simultaneously), that means we need to reset the changes made to the other rows and perform the same calculation
            again after swapping the rows. But even then, the swapped row cannot be guaranteed to have a non-zero value :(

        Solution A1: 
            Serialization before parallelism. Have one rank that owns the target row calculate to see if it would have
            a zero leading principal minor. 
            If true, swap and rerun and check again.
            If false, all ranks run.
    */

}