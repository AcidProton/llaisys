#include "../../include/llaisys/models/qwen2.h"
#include <cstdlib>

int main(){
    LlaisysQwen2Meta meta;
    meta.dh=128;
    meta.di=8960;
    meta.dtype=LLAISYS_DTYPE_BF16;
    meta.end_token=151643;
    meta.epsilon=1e-06;
    meta.hs=1536;
    meta.maxseq=131072;
    meta.nh=12;
    meta.nkvh=2;
    meta.nlayer=28;
    meta.theta=10000;
    meta.voc=151936;
    LlaisysQwen2Model* model = llaisysQwen2ModelCreate(&meta,llaisysDeviceType_t::LLAISYS_DEVICE_CPU,0);
    LlaisysQwen2Weights* w = llaisysQwen2ModelWeights(model);
    LlaisysQwen2Weights weights = *w;
    void* data1 = malloc(2*meta.voc*meta.hs);
    void* data2 = malloc(2*meta.voc*meta.hs);
    void* data3 = malloc(2*meta.hs);
    tensorLoad(weights.in_embed, data1);
    tensorLoad(weights.out_embed, data2);
    tensorLoad(weights.out_norm_w, data1);
    return 0;
}