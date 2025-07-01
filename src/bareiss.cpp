#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <vector>
#include <random>
#include <iostream>
#include <sstream>

int main(int argc, char** argv){

    ygm::comm world(&argc, &argv);

    static ygm::comm& s_world = world;

    ygm::container::map<long long, std::vector<long long>> matrix(world);

    static int n = 30;

    /*
        NOTE: since all ranks needs access to the first row,
              I decided to let all ranks own that row. It seemed
              inefficient for all ranks to constantly communicate
              to the rank that owns it.
    */
    

    // add the rows
    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed with the rank number
    std::uniform_int_distribution<> random_n(1, 10);

    if(world.rank0()){
        for(long long i = 0; i < n; i++){
            std::vector<long long> row;
            for(long long j = 0; j < n; j++){
                row.push_back(random_n(gen));
            }
            matrix.async_insert(i, row);
            std::ostringstream oss;
            for(long long num : row){
                oss << num << " ";
            }
            std::cout << oss.str() << std::endl;
        }
        // std::vector<long long> v1 = {2, 8, 9, 8, 2, 10, 10, 10};
        // std::vector<long long> v2 = {6, 2, 7, 8, 8, 6, 9, 8};
        // std::vector<long long> v3 = {2, 3, 3, 8, 3, 1, 8, 2};
        // std::vector<long long> v4 = {5, 10, 7, 9, 8, 9, 4, 5};
        // std::vector<long long> v5 = {9, 2, 2, 6, 3, 3, 8, 2};
        // std::vector<long long> v6 = {3, 2, 8, 1, 9, 3, 10, 7};
        // std::vector<long long> v7 = {3, 2, 2, 2, 9, 10, 5, 10};
        // std::vector<long long> v8 = {7, 2, 10, 7, 10, 3, 8, 2};
        // matrix.async_insert(0, v1);
        // matrix.async_insert(1, v2);
        // matrix.async_insert(2, v3);
        // matrix.async_insert(3, v4);
        // matrix.async_insert(4, v5);
        // matrix.async_insert(5, v6);
        // matrix.async_insert(6, v7);
        // matrix.async_insert(7, v8);
    }

    world.barrier();


    double start = MPI_Wtime();

    static std::vector<std::pair<long long, std::vector<long long>>> gathered_rows;
    static bool flipSign = false;

    for(static long long k = 0; k < n - 1; k++){ // k < n - 1

        matrix.gather(gathered_rows);
        std::sort(gathered_rows.begin(), gathered_rows.end());
        static long long divisor;

        if(k == 0){
            divisor = 1;
        }
        else{
            // M[k - 1, k - 1]
            divisor = gathered_rows.at(k - 1).second.at(k - 1);
        }

        bool Bareiss_safe = true;

        if(k < n - 2){ // the pivot in this row is the determinant and it can be zero. its fine.
            if(world.rank0()){
                // the future pivot M(k + 1, k + 1) of the next iteration k++

                long long pivot = (gathered_rows.at(k + 1).second.at(k + 1) * gathered_rows.at(k).second.at(k) - 
                            gathered_rows.at(k + 1).second.at(k) * gathered_rows.at(k).second.at(k + 1)) 
                            / divisor;
                long long zero_key = gathered_rows.at(k + 1).first;
                std::vector<long long> zero_vector = gathered_rows.at(k + 1).second;

                long long non_zero_key;
                std::vector<long long> non_zero_vector;

                if(pivot == 0){ // need to swap 
                    
                    world.cout0("need to swap");
                    for(long long i = k + 2; i < n; i++){
                        
                        pivot = (gathered_rows.at(i).second.at(k + 1) * gathered_rows.at(k).second.at(k) - 
                            gathered_rows.at(i).second.at(k) * gathered_rows.at(k).second.at(k + 1)) 
                            / divisor;
                        
                        long long a = gathered_rows.at(i).second.at(k + 1);  // M[i][k+1]
                        long long b = gathered_rows.at(k).second.at(k);      // M[k][k]
                        long long c = gathered_rows.at(i).second.at(k);      // M[i][k]
                        long long d = gathered_rows.at(k).second.at(k + 1);  // M[k][k+1]

                        long long numerator = a * b - c * d;
                        pivot = numerator / divisor;

                        // Prlong long each factor
                        std::cout << "M[i][k+1] (a) = " << a << std::endl;
                        std::cout << "M[k][k]   (b) = " << b << std::endl;
                        std::cout << "M[i][k]   (c) = " << c << std::endl;
                        std::cout << "M[k][k+1] (d) = " << d << std::endl;
                        std::cout << "Numerator = a*b - c*d = " << a << "*" << b << " - " << c << "*" << d << " = " << numerator << std::endl;
                        std::cout << "Divisor = " << divisor << std::endl;
                        std::cout << "Pivot = " << pivot << std::endl;
                        
                        if(pivot != 0){
                            non_zero_key = gathered_rows.at(i).first;
                            non_zero_vector = gathered_rows.at(i).second;
                            break;
                        }
                    }

                    if(pivot == 0){
                        Bareiss_safe = false;
                    }

                    matrix.async_erase(zero_key);
                    matrix.async_erase(non_zero_key);
                    std::cout << "swapping row " << zero_key << " and " << non_zero_key << std::endl; 

                    matrix.async_insert(zero_key, non_zero_vector);
                    matrix.async_insert(non_zero_key, zero_vector);


                    if(!flipSign){
                        flipSign = true;
                    }
                    else{
                        flipSign = false;
                    }
                }


            }

            world.barrier();

            if(!Bareiss_safe){

                if(world.rank0()){
                    std::cout << "-----------------------------------" << std::endl;
                    for(std::pair<long long, std::vector<long long>> key_value : gathered_rows){
                        std::ostringstream oss;
                        for(long long num : key_value.second){
                            oss << num << ", ";
                        }
                        std::cout << oss.str() << std::endl;
                    }
                }
                world.async_bcast([](){
                    s_world.cout("the matrix is invalid");
                    std::exit(EXIT_FAILURE);
                });
            }

            gathered_rows.clear();
            matrix.gather(gathered_rows);
            std::sort(gathered_rows.begin(), gathered_rows.end());

        }

        

        // since the rank that owns row k has its local_row filled, empty rows will get filtered out
        // ISSUE: local_row is not populated by the time this run since it is asynchronous 
        // shared_row = ygm::all_reduce(local_row, [](std::vector<long long> a, std::vector<long long> b) {
        //     return a.empty() ? b : a; 
        // }, world);
        
        world.barrier();

        matrix.for_all([&](long long row_num, std::vector<long long>& row){
            if(row_num >= k + 1){
                for(long long j = k + 1; j < n; j++){   
                    long long prev_num = row.at(j);
                    
                    if(row_num == k + 1 && j == k + 1){
                        std::cout << "current iteration k: " << k << std::endl;
                        std::cout << "("
                                << row.at(j) << " * " << gathered_rows.at(k).second.at(k)
                                << " - "
                                << row.at(k) << " * " << gathered_rows.at(k).second.at(j)
                                << ") / " << divisor
                                << std::endl;

                    }

                    row.at(j) = (row.at(j) * gathered_rows.at(k).second.at(k) - row.at(k) * gathered_rows.at(k).second.at(j)) / divisor;
                    // std::cout << prev_num << " -> " << row.at(j) << " at position " << row_num << ", " << j << "." << std::endl;

                    if(row_num == k + 1 && j == k + 1){
                        std::cout << "stored " << row.at(j) << std::endl;
                    }


                    
                }

                row.at(k) = 0;
            }
        });


  

        gathered_rows.clear();

        world.barrier();
    }

    double end = MPI_Wtime();

    if (world.rank0()) {
        std::cout << "Elapsed time: " << (end - start) << " seconds\n";
    }

    if(world.rank0()){
        matrix.async_visit(n - 1, [](long long row_num, std::vector<long long>& row, bool flip){
            if(flip){
                row.at(row.size() - 1) = -1 * row.at(row.size() - 1);
            }
        }, flipSign);
    }

    gathered_rows.clear();
    matrix.gather(gathered_rows);
    std::sort(gathered_rows.begin(), gathered_rows.end());

    if(world.rank0()){
        std::cout << "-------------------" << std::endl;
        for(std::pair<long long, std::vector<long long>> key_value : gathered_rows){
            std::ostringstream oss;
            for(long long num : key_value.second){
                oss << num << ", ";
            }
            std::cout << oss.str() << std::endl;
        }
    }

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

    /*
        implement ygm::array?

        copying the entire matrix is inefficient, thus instead use asynchronous communication with 
        barrier to ensure that all asynchronous operations are completed
        implement barrier
        

    */

}