#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out,in,pos_ids);
    CHECK_SAME_DTYPE(out->dtype(),in->dtype());
    CHECK_SAME_DTYPE(llaisysDataType_t::LLAISYS_DTYPE_I64,pos_ids->dtype());
    CHECK_SAME_SHAPE(3,out->ndim());
    CHECK_SAME_SHAPE(out->shape(),in->shape());
    CHECK_SAME_SHAPE(1,pos_ids->ndim());
    ASSERT(in->isContiguous(),"Argmax: all input must be contiguous");
    ASSERT(pos_ids->isContiguous(),"Argmax: all pos_id must be contiguous"); 
    size_t seq_len = in->shape()[0];
    size_t nhead = in->shape()[1];
    size_t d = in->shape()[2];
    CHECK_SAME_SHAPE(pos_ids->shape()[0],seq_len);

    // always support cpu calculation
    if (in->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(), theta, seq_len, nhead, d);
    }

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), in->dtype(), theta, seq_len, nhead, d);
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
