#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out,in,weight,bias);
    CHECK_SAME_DTYPE(out->dtype(),in->dtype(),weight->dtype(),bias->dtype());
    CHECK_SAME_SHAPE(2,out->ndim());
    CHECK_SAME_SHAPE(2,in->ndim());
    CHECK_SAME_SHAPE(2,weight->ndim());
    CHECK_SAME_SHAPE(1,bias->ndim());
    ASSERT(in->isContiguous(),"Argmax: all input must be contiguous");
    ASSERT(weight->isContiguous(),"Argmax: all weight must be contiguous");
    ASSERT(bias->isContiguous(),"Argmax: all bias must be contiguous");
    size_t k = in->shape()[1];
    size_t m = in->shape()[0];
    size_t n = weight->shape()[0];
    CHECK_SAME_SHAPE(k,weight->shape()[1]);
    CHECK_SAME_SHAPE(n,weight->shape()[0]);

    // always support cpu calculation
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), weight->dtype(), m, n, k);
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), weight->dtype(), m, n, k);
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
