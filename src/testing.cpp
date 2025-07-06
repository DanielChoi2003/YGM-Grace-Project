#include <ygm/comm.hpp>


int main(int argc, char** argv){
    static int messages_received;
    
    ygm::comm world(&argc, &argv);
    messages_received = 0;

    for (int i = 0; i < world.rank(); ++i) {
        world.async(i, []() { ++messages_received; });
    }

    world.local_wait_until([&world]() { return messages_received == world.size() - world.rank() - 1; });


    world.cout("finished");
    return 0;
}