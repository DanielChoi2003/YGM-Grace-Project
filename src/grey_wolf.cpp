#include <ygm/comm.hpp>
#include <ygm/container/map.hpp>
#include <ygm/container/bag.hpp>
#include <vector>
#include <math.h>
#include <string>
#include <queue>
#include <random>

struct wolf_info{

    template <class Archive>
    // what does this do?
    // i assume this determines what data is package when it is sent to another rank
    void serialize( Archive & ar )
    {
        ar(fitness_value, pos);
    }

    std::vector<double> pos;
    double fitness_value;
};

double fitness_equation(std::vector<double> position){

    double x1 = position[0];
    double x2 = position[1];
    double result = pow(x1, 2) - x1 * x2 + 
                    pow(x2, 2) + 2 * x1 + 4 * x2 + 3;

    return result;
}


double random01(){
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

std::vector<double> gen_random_vector(int lower, int upper){
    double x1 = lower + random01() * (upper - lower);
    double x2 = lower + random01() * (upper - lower);

    return {x1, x2};
}


std::vector<double> calculate_X(std::vector<double> alpha_position, std::vector<double> current_position,
                    double decrementing_a){

    double A = 2 * decrementing_a * random01() - decrementing_a;

    double C = 2 * random01();

    std::vector<double> alpha_distance;
    double x = abs(C * alpha_position[0] - current_position[0]);
    double y = abs(C * alpha_position[1] - current_position[1]);
    alpha_distance.push_back(x);
    alpha_distance.push_back(y);

    std::vector<double> new_X;
    new_X.push_back(alpha_position[0] - A * alpha_distance[0]);
    new_X.push_back(alpha_position[1] - A * alpha_distance[1]);

    return new_X;
}
// return the best value
// map the vector to the result of objective function
// find the 3 best results and return the corresponding vectors

/* 
    the issue is how to collect all the fitness values from ranks
    and compare and assign the top 3 "inputs"

    calculating fitness values can be done independently by each rank, but
    the issue is that we have to collect all the fitness values and each rank's map.
    then a single rank calls this function
*/

using map_type = ygm::container::map<int, wolf_info>;

void add_wolf(map_type& wolf_map,
              int wolf_num, std::vector<double> position) {

  auto inserter = [](int wolf_num, wolf_info& wi, std::vector<double> position) {
    wi.pos = position;
  };

  wolf_map.async_visit(wolf_num, inserter, position);
}

int main(int argc, char** argv){

    ygm::comm world(&argc, &argv);

    static ygm::comm& s_world = world;

    map_type wolf_map(world);



    // STEP 1: initialize the population and the iteration size
    static double lower = -5;
    static double upper = 5;

    if(world.rank0()){
        for(int i = 0; i < 6; i++){
            std::vector<double> vec = gen_random_vector(lower, upper);
            add_wolf(wolf_map, i, vec);
        }
    }



    static int global_t = 0;

    static int alpha;
    static int beta;
    static int delta;

    // STEP 2: find the best 3 solutions (alpha, beta, delta)
    /*
        each rank will propose their own top 3 solutions.
            For this, we need an objective function to determine what are the best 3 solutions
            (fitness equation)
        Then use all-gather to collect those best 3 solution samples
        After collecting all, let rank 0 to do the calculation of global top 3.
            It is serial, but i don't think it is too inefficient
    */

    std::vector<double> fittest_solution;

   while(global_t <= 5000){

        auto cmp = [](const std::pair<int, wolf_info> &a, const std::pair<int, wolf_info> &b) {
            return a.second.fitness_value < b.second.fitness_value;  
        };

        // local best wolves ranked by minimum value of fitness value (lower the fitness value, the better)
        // static std::priority_queue <wolf_info, std::vector<wolf_info>, decltype(cmp)> best_wolves(cmp);


        wolf_map.for_all([](int wolf_num, wolf_info& wi){
            wi.fitness_value = abs(fitness_equation(wi.pos));
        });


        // send the top three to the rank 0


        //wolf_map.gather_topk(rank0_candidate_wolves, 3, cmp, 0);
        static std::vector<std::pair<int, wolf_info>> top_three;
        top_three = wolf_map.gather_topk(3, cmp);


        /*
            A = 2a.r₁ - a
            C = 2.r₂ 
            D = |C.Xₚ(t) - X(t)|
            Dₐₗₚₕₐ = |C₁.Xₐₗₚₕₐ - X|
            X₁ = Xₐₗₚₕₐ - a₁.(Dₐₗₚₕₐ)
        

            calculate the "X" for alpha, delta, and beta for each wolf

            then calculate every wolf's new position by averaging the 3 X's
        */

        // slowly decrements from 2 to 0
        static double a = 2 - 2 * ((double)global_t / 200);

        wolf_map.for_all([](int wolf_num, wolf_info& wi){
            std::vector<double> alpha_X = calculate_X(top_three[0].second.pos, wi.pos, a);
            std::vector<double> beta_X = calculate_X(top_three[1].second.pos, wi.pos, a);
            std::vector<double> delta_X = calculate_X(top_three[2].second.pos, wi.pos, a);

            std::vector<double> new_X;
            new_X.push_back((alpha_X[0] + beta_X[0] + delta_X[0]) / 3);
            new_X.push_back((alpha_X[1] + beta_X[1] + delta_X[1]) / 3);


            wi.pos = new_X;
        });

        world.barrier();
    // synchronize here

        global_t++;

        fittest_solution = top_three[0].second.pos;
        
   }

   world.barrier();


   world.cout0("before printing");
   world.cout0(fittest_solution[0], ", ", fittest_solution[1]);
   world.cout0(abs(fitness_equation(fittest_solution)));

   return 0;
}





/*
    D = |C.Xₚ(t) - X(t)|
    
    * D is the distance between the Prey and the Grey Wolf
    * Xₚ(t) is the location of the Prey
    * X(t) is the location of the Grey Wolf
    
    X(t + 1) = Xₚ(t) - A.D

    * The Grey Wolf's next position is determined with: 
        Current Prey's position minus A.D, where A.D is the 
        pairwise multiplication of A and D

    A = 2a.r₁ - a

    * r₁ is a random number
    * a linearly decreases from 2 to 0, which allows the algorithm to
        slowly converge to one solution   


    C = 2.r₂ 
    * r₂ is also a random number

    Dₐₗₚₕₐ = |C₁.Xₐₗₚₕₐ - X|
    * calculate the distance to Alpha Grey Wolf

    X₁ = Xₐₗₚₕₐ - a₁.(Dₐₗₚₕₐ)
    * position that I'm supposed to be in if I follow the Alpha

    * Do this for Beta and Delta

    X = (X₁ + X₂ + X₃) / 3
    * position that I'm supposed to be in if I follow all Alpha, Beta, Delta
    * This allows you to get closer to the "Prey"
    * 


    In the context of this algorithm, we don't know the location of the Prey
    So we are assuming that based on the locations of the 3 best solutions (
    Alpha, Beta, Delta), we can get pretty close to the most optimal solution (Prey)


    -1 < A < 1
    A > 1 or A < -1

    * We need to address scenarios of "what if the 3 'best' solutions are
        locally stuck?". That means we need those 3 to get out of the bubble,
        and A allows us to do that.

        Remember the equation A = 2a.r₁ - a, where r₁ is a random value and 
        a is a linearly decrementing value as t approaches its end value
        Due to r₁, A is essentially a random value with a range of 
        [-a, a]. 

        So when -1 < A < 1, the wolf is converging towards the prey
        when A > 1 or A < -1, the wolf is diverging away from the prey

    
    Grey wolves search for the prey according to the positions of Alpha, Beta, and Delta

    I believe that "a" is responsible for converging and "A" is responsible for diverging
    and r₁ plays a role in randomizing the occurrence of divergence

    "A" eventually converges towards zero since "a" also converges towards zero (
    meaning that closer to the last iterations), and that 
    convergence to one point may not be the best solution. So...
    We need another mechanism to avoid getting stuck (make it more random).
        parameter C does exactly that. Remember the equation C = 2.r₂
        It is completely random. 
        It emphasizes / focues on the prey when C > 1
        It deemphasizes on the prey when C < 1

        This allows Grey Wolf Opt. to show a more random behabior throughout
        optimization to encourage exploration and avoid local optima.
*/