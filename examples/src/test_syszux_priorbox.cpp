#include "syszux_priorbox.h"

using namespace std;
int main(int argc, const char* argv[]) {
    gemfield_org::PriorBox pb({{16,32},{64,128},{256,512}}, {8,16,32});
    vector<int> img_size = {224,312};
    auto x = pb.forward(img_size);
    std::cout<<x<<std::endl;
    return 0;
}