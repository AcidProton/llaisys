#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out,index,weight);
    CHECK_SAME_DTYPE(out->dtype(),weight->dtype());
    CHECK_SAME_DTYPE(llaisysDataType_t::LLAISYS_DTYPE_I64,index->dtype());
    CHECK_SAME_SHAPE(2,weight->ndim());
    CHECK_SAME_SHAPE(1,index->ndim());
    CHECK_SAME_SHAPE(2,out->ndim());
    ASSERT(index->isContiguous(),"Argmax: all index must be contiguous");
    ASSERT(weight->isContiguous(),"Argmax: all weight must be contiguous");
    size_t embedding_dim = weight->shape()[1];
    size_t index_size = index->shape()[0];
    CHECK_SAME_SHAPE(index_size,out->shape()[0]);
    CHECK_SAME_SHAPE(embedding_dim,out->shape()[1]);

    // always support cpu calculation
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), embedding_dim, index_size);
    }

    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), weight->dtype(), embedding_dim, index_size);
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
