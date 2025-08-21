#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val,q,k,v);
    CHECK_SAME_DTYPE(attn_val->dtype(),q->dtype(),k->dtype(),v->dtype());
    CHECK_SAME_SHAPE(3,attn_val->ndim(),q->ndim(),k->ndim(),v->ndim());
    ASSERT(q->isContiguous(),"Argmax: all querys must be contiguous");
    ASSERT(k->isContiguous(),"Argmax: all keys must be contiguous"); 
    ASSERT(v->isContiguous(),"Argmax: all values must be contiguous");
    size_t seqlen = q->shape()[0];
    size_t nhead = q->shape()[1];
    size_t d = q->shape()[2];
    size_t total_len = v->shape()[0];
    size_t nkvhead = v->shape()[1];
    size_t dv = v->shape()[2];
    CHECK_SAME_SHAPE(seqlen,attn_val->shape()[0]);
    CHECK_SAME_SHAPE(nhead,attn_val->shape()[1]);
    CHECK_SAME_SHAPE(dv,attn_val->shape()[2]);
    CHECK_SAME_SHAPE(total_len,k->shape()[0]);
    CHECK_SAME_SHAPE(nkvhead,k->shape()[1]);
    CHECK_SAME_SHAPE(d,v->shape()[2]);

    // always support cpu calculation
    if (q->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(), 
            scale, seqlen, nhead, d, total_len, nkvhead, dv);
    }

    llaisys::core::context().setDevice(q->deviceType(), q->deviceId());

    switch (q->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), q->dtype(), 
            scale, seqlen, nhead, d, total_len, nkvhead, dv);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
