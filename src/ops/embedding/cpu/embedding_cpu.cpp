#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t embedding_dim, size_t index_size) {
    for (size_t i = 0; i < index_size; i++) {
        int64_t token_id = index[i];
        const T *src_row = weight + token_id * embedding_dim;
        T *dst_row = out + i * embedding_dim;
        std::memcpy(dst_row, src_row, embedding_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, std::byte *index, const std::byte *weight, llaisysDataType_t type, size_t embedding_dim, size_t index_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const float *>(weight),
                    embedding_dim, index_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::bf16_t *>(weight),
                    embedding_dim, index_size);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index), reinterpret_cast<const llaisys::fp16_t *>(weight),
                    embedding_dim, index_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu