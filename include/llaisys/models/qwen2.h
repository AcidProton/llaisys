#ifndef LLAISYS_MODELS_QWEN2_H
#define LLAISYS_MODELS_QWEN2_H

#include "../tensor.h"

__C {
    typedef struct LlaisysQwen2Meta {
        llaisysDataType_t dtype;
        size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc; // num_hidden_layers, hidden_size, num_attention_heads, num_key_value_heads, head_dim, intermediate_size, max_position_embeddings, vocab_size
        float epsilon, theta;
        int64_t end_token;
    } LlaisysQwen2Meta;

    typedef struct LlaisysQwen2Weights {
        llaisysTensor_t in_embed;
        llaisysTensor_t out_embed;
        llaisysTensor_t out_norm_w;   // a.k.a. model.norm.weight
        llaisysTensor_t *attn_norm_w; // a.k.a. input_layernorm.weight
        llaisysTensor_t *attn_q_w;
        llaisysTensor_t *attn_q_b;
        llaisysTensor_t *attn_k_w;
        llaisysTensor_t *attn_k_b;
        llaisysTensor_t *attn_v_w;
        llaisysTensor_t *attn_v_b;
        llaisysTensor_t *attn_o_w;
        llaisysTensor_t *mlp_norm_w; // a.k.a. post_attention_layernorm.weight
        llaisysTensor_t *mlp_gate_w;
        llaisysTensor_t *mlp_up_w;
        llaisysTensor_t *mlp_down_w;
    } LlaisysQwen2Weights;

    typedef struct LlaisysQwen2Activation {
        llaisysTensor_t tokens; //[seqlen]
        llaisysTensor_t pos_ids; //[seqlen]
        llaisysTensor_t in_embed; //[seqlen,hidden_size]
        llaisysTensor_t attn_residual;//[seqlen,hidden_size]
        llaisysTensor_t attn_norm;//[seqlen,hidden_size]
        llaisysTensor_t attn_q;//[seqlen,nh,dh]
        llaisysTensor_t attn_k;//[seqlen,nkvh,dh]
        llaisysTensor_t attn_q_pos;//[seqlen,nh,dh]
        llaisysTensor_t attn_val;//[seqlen,nh,dh]
        llaisysTensor_t attn_o;//[seqlen,hidden_size]
        llaisysTensor_t mlp_residual;//[seqlen,hidden_size]
        llaisysTensor_t mlp_norm;//[seqlen,hidden_size]
        llaisysTensor_t mlp_gate;//[seqlen,di]
        llaisysTensor_t mlp_active;//[seqlen,di]
        llaisysTensor_t mlp_up;//[seqlen,di]
        llaisysTensor_t mlp_down;//[seqlen,hidden_size]
        llaisysTensor_t mlp_out;//[seqlen,hidden_size]
        llaisysTensor_t out_norm;//[1,hidden_size]
        llaisysTensor_t out_token_val;//[1,voc]
        llaisysTensor_t max_token_val;//[1,]
        llaisysTensor_t max_token_ids;//[1,]
    } LlaisysQwen2Activation;

    typedef struct LlaisysQwen2KVcache {
        llaisysTensor_t *attn_v;//layer*[total_len,nkvh,dh] (cache)
        llaisysTensor_t *attn_k_pos;//layer*[total_len,nkvh,dh] (cache)
    } LlaisysQwen2KVcache;
    

    typedef struct LlaisysQwen2Model {  
        LlaisysQwen2Meta *meta;
        llaisysDeviceType_t device;
        int device_ids;
        LlaisysQwen2Weights *weights;
    } LlaisysQwen2Model;

    typedef struct LlaisysQwen2Context {
        size_t total_len;
        size_t seqlen;
        LlaisysQwen2Activation *activation;
        LlaisysQwen2KVcache *kvcache;
    } LlaisysQwen2Context;

    __export LlaisysQwen2Model* llaisysQwen2ModelCreate(LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int device_ids);

    __export void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model);

    __export LlaisysQwen2Weights* llaisysQwen2ModelWeights(LlaisysQwen2Model *model);

    __export llaisysTensor_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t * token_ids, size_t ntoken, size_t max_new_token);
}
#endif // LLAISYS_MODELS_QWEN2_H
