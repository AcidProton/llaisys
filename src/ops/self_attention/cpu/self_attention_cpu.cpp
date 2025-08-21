#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale, size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
    // attn_val [seqlen, nhead, dv]
    // q [seqlen, nhead, d]
    // k [total_len, nkvhead, d]
    // v [total_len, nkvhead, dv]
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        // q @ k.T = [seqlen, nhead, total_len]
        std::vector<float> attn(seqlen*nhead*total_len);
        for(size_t q_seq=0; q_seq<seqlen; q_seq++){
            for(size_t q_head=0; q_head<nhead; q_head++){
                size_t mapped_k_head = q_head / nkvhead;
                float max_val = -1e30f;
                for(size_t k_len=0; k_len<total_len; k_len++){
                    float sum = 0;
                    for(size_t dim=0; dim<d; dim++){
                        sum += llaisys::utils::cast<float>(q[q_seq*nhead*d + q_head*d + dim]) * llaisys::utils::cast<float>(k[k_len*nkvhead*d + mapped_k_head*d + dim]);
                    }
                    float val = scale * sum;
                    attn[q_seq*nhead*total_len + q_head*total_len + k_len] = val;
                    max_val = std::max(max_val,val);
                }
                for(size_t k_len=0; k_len<total_len; k_len++){
                    attn[q_seq*nhead*total_len + q_head*total_len + k_len] -= max_val;
                }
            }
        }
        // casual_softmax
        for(size_t seq=0; seq<seqlen; seq++){
            for(size_t head=0; head<nhead; head++){
                float sum =0;
                for(size_t len=0; len<total_len-seqlen+1+seq; len++){
                    sum += exp(attn[seq*nhead*total_len + head*total_len + len]);
                }
                for(size_t len=0; len<total_len; len++){
                    if(len < total_len-seqlen+1+seq){
                        float ele = exp(attn[seq*nhead*total_len + head*total_len + len]);
                        attn[seq*nhead*total_len + head*total_len + len] = ele / sum;
                    }else{
                        attn[seq*nhead*total_len + head*total_len + len] = 0;
                    }
                }
            }
        }
        // @ v = [seqlen, nhead, dv]
        for(size_t seq=0; seq<seqlen; seq++){
            for(size_t head=0; head<nhead; head++){
                size_t mapped_v_head = head / nkvhead;
                for(size_t d=0; d<dv; d++){
                    float sum = 0;
                    for(size_t len=0; len<total_len; len++){
                        sum += attn[seq*nhead*total_len + head*total_len + len] * llaisys::utils::cast<float>(v[len*nkvhead*dv + mapped_v_head*dv + d]);
                    }
                    attn_val[seq*nhead*dv + head*dv + d] = llaisys::utils::cast<T>(sum);
                }
            }
        }
    } else {
        // q @ k.T = [seqlen, nhead, total_len]
        std::vector<float> attn(seqlen*nhead*total_len);
        for(size_t q_seq=0; q_seq<seqlen; q_seq++){
            for(size_t q_head=0; q_head<nhead; q_head++){
                size_t mapped_k_head = q_head / nkvhead;
                float max_val = -1e30f;
                for(size_t k_len=0; k_len<total_len; k_len++){
                    float sum = 0;
                    for(size_t dim=0; dim<d; dim++){
                        sum += q[q_seq*nhead*d + q_head*d + dim] * k[k_len*nkvhead*d + mapped_k_head*d + dim];
                    }
                    float val = scale * sum;
                    attn[q_seq*nhead*total_len + q_head*total_len + k_len] = val;
                    max_val = std::max(max_val,val);
                }
                for(size_t k_len=0; k_len<total_len; k_len++){
                    attn[q_seq*nhead*total_len + q_head*total_len + k_len] -= max_val;
                }
            }
        }
        // casual_softmax
        for(size_t seq=0; seq<seqlen; seq++){
            for(size_t head=0; head<nhead; head++){
                float sum =0;
                for(size_t len=0; len<total_len-seqlen+1+seq; len++){
                    sum += exp(attn[seq*nhead*total_len + head*total_len + len]);
                }
                for(size_t len=0; len<total_len; len++){
                    if(len < total_len-seqlen+1+seq){
                        float ele = exp(attn[seq*nhead*total_len + head*total_len + len]);
                        attn[seq*nhead*total_len + head*total_len + len] = ele / sum;
                    }else{
                        attn[seq*nhead*total_len + head*total_len + len] = 0;
                    }
                }
            }
        }
        // @ v = [seqlen, nhead, dv]
        for(size_t seq=0; seq<seqlen; seq++){
            for(size_t head=0; head<nhead; head++){
                size_t mapped_v_head = head / nkvhead;
                for(size_t d=0; d<dv; d++){
                    float sum = 0;
                    for(size_t len=0; len<total_len; len++){
                        sum += attn[seq*nhead*total_len + head*total_len + len] * v[len*nkvhead*dv + mapped_v_head*dv + d];
                    }
                    attn_val[seq*nhead*dv + head*dv + d] = sum;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, llaisysDataType_t type, 
        float scale, size_t seqlen, size_t nhead, size_t d, size_t total_len, size_t nkvhead, size_t dv) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
            scale, seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k), 
        reinterpret_cast<const llaisys::bf16_t *>(v), scale, seqlen, nhead, d, total_len, nkvhead, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k), 
        reinterpret_cast<const llaisys::fp16_t *>(v), scale, seqlen, nhead, d, total_len, nkvhead, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu