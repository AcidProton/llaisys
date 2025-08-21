#pragma once
#include "llaisys.h"

#include <cstddef>
#include <vector>

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, 
        float scale, size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv);
}