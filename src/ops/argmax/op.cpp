#include "op.hpp"


#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx,max_val,vals);
    CHECK_SAME_SHAPE(max_idx->shape(),max_val->shape());
    CHECK_SAME_DTYPE(max_val->dtype(),vals->dtype());
    CHECK_SAME_SHAPE(1,max_idx->ndim(),max_val->ndim());
    ASSERT(vals->isContiguous(),"Argmax: all vals must be contiguous");
    size_t rowele = vals->shape()[vals->ndim()-1];
    size_t batch = vals->numel()/rowele;
    CHECK_SAME_SHAPE(batch,max_idx->shape()[0],max_val->shape()[0]);
    // always support cpu calculation
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), rowele, batch);
    }

    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), rowele, batch);
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
