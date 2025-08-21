#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seq_len, size_t nhead, size_t d) {
    
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        for(size_t i=0; i<seq_len; i++){
            for(size_t h=0; h<nhead; h++){
                for(size_t j=0; j<d/2; j++){
                    float phi = pos_ids[i] / pow(theta,2.0*j/d);
                    float a = llaisys::utils::cast<float>(in[i*nhead*d + h*d + j]);
                    float b = llaisys::utils::cast<float>(in[i*nhead*d + h*d + d/2 + j]);
                    float sin_val = sin(phi);
                    float cos_val = cos(phi);
                    out[i*nhead*d + h*d + j] = llaisys::utils::cast<T>(a*cos_val - b*sin_val);
                    out[i*nhead*d + h*d + d/2 + j] = llaisys::utils::cast<T>(b*cos_val + a*sin_val);
                }
            }
        }
    } else {
        for(size_t i=0; i<seq_len; i++){
            for(size_t h=0; h<nhead; h++){
                for(size_t j=0; j<d/2; j++){
                    double phi = pos_ids[i] / pow(theta, 2.0*j/d);;
                    float a = in[i*nhead*d + h*d + j];
                    float b = in[i*nhead*d + h*d + d/2 +j];
                    float sin_val = sin(phi);
                    float cos_val = cos(phi);
                    out[i*nhead*d + h*d + j] = a*cos_val - b*sin_val;
                    out[i*nhead*d + h*d + d/2 + j] = b*cos_val + a*sin_val;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, float theta, size_t seq_len, size_t nhead, size_t d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, nhead, d);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, nhead, d);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, nhead, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu