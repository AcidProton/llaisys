#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out,in,weight);
    CHECK_SAME_DTYPE(out->dtype(),in->dtype(),weight->dtype());
    CHECK_SAME_SHAPE(2,out->ndim());
    CHECK_SAME_SHAPE(2,in->ndim());
    CHECK_SAME_SHAPE(1,weight->ndim());
    ASSERT(in->isContiguous(),"Argmax: all input must be contiguous");
    ASSERT(weight->isContiguous(),"Argmax: all weight must be contiguous"); 
    size_t row_size = in->shape()[1];
    size_t batch = in->shape()[0];
    CHECK_SAME_SHAPE(weight->shape()[0],row_size);
    CHECK_SAME_SHAPE(in->shape(),out->shape());

    // always support cpu calculation
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), weight->dtype(), eps, row_size, batch);
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), weight->dtype(), eps, row_size, batch);
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
