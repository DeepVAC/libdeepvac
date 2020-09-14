/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#include "syszux_nms.h"
#include <torch/script.h>

int main(int argc, const char* argv[]) {
    float threshold = 0.95;
    torch::Tensor dets_tensor;
    float dets[10][5] = {
        {608.869629,131.506607,726.548950,276.543457,0.999467671},
        {608.558289,132.701614,724.578796,275.906860,0.999419928},
        {608.443909,128.678253,726.312439,278.324097,0.916760027},
        {609.709534,129.584427,724.541199,273.925568,0.848726749},
        {609.766479,130.341476,726.441101,274.966766,0.747644246},
        {608.292908,123.771599,725.066101,276.210419,0.707930803},
        {609.070374,126.515434,723.639038,276.863739,0.649771333},
        {607.130554,132.601349,724.788513,279.786499,0.600724220},
        {608.286377,133.966751,727.646606,279.250275,0.523717880},
        {608.372070,129.631256,724.156494,277.742035,0.497736454}
    };
    dets_tensor = torch::from_blob(dets, {10, 5});
    auto keep = gemfield_org::nms(dets_tensor, threshold);
    std::cout << keep << std::endl;
    return 0;
}
