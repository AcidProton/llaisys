#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out,gate,up);
    CHECK_SAME_DTYPE(out->dtype(),gate->dtype(),up->dtype());
    CHECK_SAME_SHAPE(out->shape(),gate->shape(),up->shape());
    ASSERT(gate->isContiguous(),"Argmax: all gate must be contiguous");
    ASSERT(up->isContiguous(),"Argmax: all up must be contiguous"); 

    // always support cpu calculation
    if (gate->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(), gate->dtype(), gate->numel());
    }

    llaisys::core::context().setDevice(gate->deviceType(), gate->deviceId());

    switch (gate->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(), gate->dtype(), gate->numel());
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
