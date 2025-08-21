#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t embedding_dim, size_t index_size);
}