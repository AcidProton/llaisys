#include "llaisys_tensor.hpp"
#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"
#include "math.h"

__C{
    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice){
        // 在device上创建tensor，分配空间
        LlaisysQwen2Model *model = new LlaisysQwen2Model;
        model->meta = meta;
        model->weights = new LlaisysQwen2Weights;
        size_t shape[] = {meta->voc,meta->hs};
        model->weights->in_embed = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
        model->weights->attn_norm_w = new llaisysTensor_t[meta->nlayer];
        model->weights->attn_q_w = new llaisysTensor_t[meta->nlayer];
        model->weights->attn_q_b = new llaisysTensor_t[meta->nlayer];
        model->weights->attn_k_w = new llaisysTensor_t[meta->nlayer];
        model->weights->attn_k_b = new llaisysTensor_t[meta->nlayer];
        model->weights->attn_v_w = new llaisysTensor_t[meta->nlayer];
        model->weights->attn_v_b = new llaisysTensor_t[meta->nlayer];
        model->weights->attn_o_w = new llaisysTensor_t[meta->nlayer];
        model->weights->mlp_norm_w = new llaisysTensor_t[meta->nlayer];
        model->weights->mlp_gate_w = new llaisysTensor_t[meta->nlayer];
        model->weights->mlp_up_w = new llaisysTensor_t[meta->nlayer];
        model->weights->mlp_down_w = new llaisysTensor_t[meta->nlayer];
        for(size_t i=0;i<meta->nlayer;i++){
            size_t shape[] = {meta->hs};
            model->weights->attn_norm_w[i]=tensorCreate(shape,1,meta->dtype,device,device_ids[0]);
            size_t shape[] = {meta->nh*meta->dh,meta->hs};
            model->weights->attn_q_w[i]=tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            size_t shape[] = {meta->nh*meta->dh};
            model->weights->attn_q_b[i]=tensorCreate(shape,1,meta->dtype,device,device_ids[0]);
            size_t shape[] = {meta->nkvh*meta->dh,meta->hs};
            model->weights->attn_k_w[i]=tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            size_t shape[] = {meta->nkvh*meta->dh};
            model->weights->attn_k_b[i]=tensorCreate(shape,1,meta->dtype,device,device_ids[0]);
            size_t shape[] = {meta->nkvh*meta->dh,meta->hs};
            model->weights->attn_v_w[i]=tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            size_t shape[] = {meta->nkvh*meta->dh};
            model->weights->attn_v_b[i]=tensorCreate(shape,1,meta->dtype,device,device_ids[0]);
            size_t shape[] = {meta->hs,meta->nh*meta->dh};
            model->weights->attn_o_w[i]=tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            size_t shape[] = {meta->hs};
            model->weights->mlp_norm_w[i]=tensorCreate(shape,1,meta->dtype,device,device_ids[0]);
            size_t shape[] = {meta->di,meta->hs};
            model->weights->mlp_up_w[i]=tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            size_t shape[] = {meta->hs,meta->di};
            model->weights->mlp_down_w[i]=tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
        }
        size_t shape[] = {meta->hs};
        model->weights->out_norm_w = tensorCreate(shape,1,meta->dtype,device,device_ids[0]);
        size_t shape[]= {meta->voc,meta->hs};
        model->weights->out_embed = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
        return model; 
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model * model){

        tensorDestroy(model->weights->in_embed);
        tensorDestroy(model->weights->out_embed);
        tensorDestroy(model->weights->out_norm_w);
        for(size_t i=0;i<model->meta->nlayer;i++){
            tensorDestroy(model->weights->attn_norm_w[i]);
            tensorDestroy(model->weights->attn_q_w[i]);
            tensorDestroy(model->weights->attn_q_b[i]);
            tensorDestroy(model->weights->attn_k_w[i]);
            tensorDestroy(model->weights->attn_k_b[i]);
            tensorDestroy(model->weights->attn_v_w[i]);
            tensorDestroy(model->weights->attn_v_b[i]);
            tensorDestroy(model->weights->attn_o_w[i]);
            tensorDestroy(model->weights->mlp_norm_w[i]);
            tensorDestroy(model->weights->mlp_up_w[i]);
            tensorDestroy(model->weights->mlp_down_w[i]);
        }
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model){
        return model->weights;
    }

    LlaisysQwen2Context *llaisysQwen2ModelCreateContext(const LlaisysQwen2Meta *meta, LlaisysQwen2Context *last_context, size_t seqlen, 
                                                        size_t max_new_token, llaisysDeviceType_t device, int *device_ids, int ndevice){
        // naive，预先分配足够大的kvcache(total_len)
        LlaisysQwen2Context *context = new LlaisysQwen2Context;
        context->activation = new LlaisysQwen2Activation;
        if(last_context==nullptr){
            context->last_pos = 0;
            context->seqlen = seqlen;
            size_t shape[]={seqlen};
            context->activation->tokens = tensorCreate(shape,1,LLAISYS_DTYPE_I64,device,device_ids[0]);
            size_t shape[]={seqlen,meta->hs};
            context->activation->in_embed = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            context->activation->attn_residual = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            context->activation->attn_norm = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            size_t shape[]={seqlen,meta->nh,meta->dh};
            context->activation->attn_q = tensorCreate(shape,3,meta->dtype,device,device_ids[0]);
            size_t shape[]={seqlen,meta->nkvh,meta->dh};
            context->activation->attn_k = tensorCreate(shape,3,meta->dtype,device,device_ids[0]);
            context->activation->attn_v = new llaisysTensor_t[meta->nlayer];
            for(size_t i=0;i<meta->nlayer;i++){
                size_t shape[]={seqlen+max_new_token,meta->nkvh,meta->dh};
                context->activation->attn_v[i] = tensorCreate(shape,3,meta->dtype,device,device_ids[0]);
            }
            size_t shape[]={seqlen,meta->nh,meta->dh};
            context->activation->attn_q_pos = tensorCreate(shape,3,meta->dtype,device,device_ids[0]);
            context->activation->attn_k_pos = new llaisysTensor_t[meta->nlayer];
            for (size_t i=0; i<meta->nlayer; i++){
                size_t shape[]={seqlen+max_new_token,meta->nh,meta->dh};
                context->activation->attn_k_pos[i] = tensorCreate(shape,3,meta->dtype,device,device_ids[0]);
            }
            size_t shape[]={seqlen,meta->nh,meta->dh};
            context->activation->attn_val = tensorCreate(shape,3,meta->dtype,device,device_ids[0]);
            size_t shape[]={seqlen,meta->hs};
            context->activation->attn_o = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            context->activation->mlp_residual = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            context->activation->mlp_norm = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            context->activation->mlp_down = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            context->activation->mlp_out = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            context->activation->out_norm = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            context->activation->out_embed = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            size_t shape[]={seqlen,meta->di};
            context->activation->mlp_gate = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            context->activation->mlp_active = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            context->activation->mlp_up = tensorCreate(shape,2,meta->dtype,device,device_ids[0]);
            size_t shape[]={seqlen};
            context->activation->max_embed = tensorCreate(shape,1,meta->dtype,device,device_ids[0]);
            context->activation->max_token = tensorCreate(shape,1,meta->dtype,device,device_ids[0]);
        }
        
    }

    void llaisysQwen2ModelDestroyContext(const LlaisysQwen2Meta *meta, LlaisysQwen2Context *context){
        tensorDestroy(context->activation->tokens);
        tensorDestroy(context->activation->in_embed);
        tensorDestroy(context->activation->attn_residual);
        tensorDestroy(context->activation->attn_norm);
        tensorDestroy(context->activation->attn_q);
        tensorDestroy(context->activation->attn_k);
        tensorDestroy(context->activation->attn_q_pos);
        tensorDestroy(context->activation->attn_val);
        tensorDestroy(context->activation->attn_o);
        tensorDestroy(context->activation->mlp_residual);
        tensorDestroy(context->activation->mlp_norm);
        tensorDestroy(context->activation->mlp_down);
        tensorDestroy(context->activation->mlp_out);
        tensorDestroy(context->activation->out_norm);
        tensorDestroy(context->activation->out_embed);
        tensorDestroy(context->activation->mlp_gate);
        tensorDestroy(context->activation->mlp_active);
        tensorDestroy(context->activation->mlp_up);
        for(size_t i=0;i<meta->nlayer;i++){
            tensorDestroy(context->activation->attn_v[i]);
            tensorDestroy(context->activation->attn_k_pos[i]);
        } 
    } 

    int64_t llaisysQwen2ModelPrefill(LlaisysQwen2Model *model, LlaisysQwen2Context *ctx, int64_t * token_ids, size_t ntoken){
        //prefil seqlen=ntoken
        tensorLoad(ctx->activation->tokens, token_ids);
        llaisysEmbedding(ctx->activation->in_embed, ctx->activation->tokens, model->weights->in_embed);
        tensorLoad(ctx->activation->attn_residual, ctx->activation->in_embed->tensor->data());
        //没有pos_idx
        for(size_t i=0;i<model->meta->nlayer;i++){
            llaisysRmsNorm(ctx->activation->attn_norm, ctx->activation->attn_residual, model->weights->attn_norm_w[i], model->meta->epsilon);
            // qkv
            size_t shape[]={ctx->seqlen*model->meta->nh,model->meta->dh};
            llaisysTensor_t attn_q_2d = tensorView(ctx->activation->attn_q,shape,2);
            llaisysLinear(attn_q_2d, ctx->activation->attn_norm, model->weights->attn_q_w[i], model->weights->attn_q_b[i]);
            size_t shape[]={ctx->seqlen*model->meta->nkvh,model->meta->dh};
            llaisysTensor_t attn_k_2d = tensorView(ctx->activation->attn_k,shape,2);
            llaisysLinear(attn_k_2d, ctx->activation->attn_norm, model->weights->attn_k_w[i], model->weights->attn_k_b[i]);
            llaisysTensor_t attn_v=tensorSlice(ctx->activation->attn_v[i], 0, 0, ctx->seqlen);
            size_t shape[]={ctx->seqlen*model->meta->nkvh,model->meta->dh};
            llaisysTensor_t attn_v_2d = tensorView(attn_v,shape,2);
            llaisysLinear(attn_v_2d, ctx->activation->attn_norm, model->weights->attn_v_w[i], model->weights->attn_v_b[i]);
            // q k rope
            llaisysROPE(ctx->activation->attn_q_pos, ctx->activation->attn_q, pos_ids, model->meta->theta);
            llaisysTensor_t attn_k_pos=tensorSlice(ctx->activation->attn_k_pos[i], 0, 0, ctx->seqlen);
            llaisysROPE(attn_k_pos, ctx->activation->attn_k, pos_ids, model->meta->theta);
            //attn
            llaisysSelfAttention(ctx->activation->attn_val, ctx->activation->attn_q_pos, attn_k_pos, attn_v, (float)(1.0/sqrt(model->meta->dh)));
            size_t shape[]={ctx->seqlen,model->meta->nh*model->meta->dh};
            llaisysTensor_t attn_val=tensorView(ctx->activation->attn_val,shape,2);
            llaisysLinear(ctx->activation->attn_o, attn_val, model->weights->attn_o_w[i], NULL); //no bias linear
            llaisysAdd(ctx->activation->mlp_residual, ctx->activation->attn_residual, ctx->activation->attn_o);
            //mlp
            llaisysRmsNorm(ctx->activation->mlp_norm, ctx->activation->mlp_residual, model->weights->mlp_norm_w[i], model->meta->epsilon);
            llaisysLinear(ctx->activation->mlp_gate, ctx->activation->mlp_norm, model->weights->mlp_gate_w[i], NULL);
            llaisysLinear(ctx->activation->mlp_up, ctx->activation->mlp_norm, model->weights->mlp_up_w[i], NULL);
            llaisysSwiGLU(ctx->activation->mlp_active, ctx->activation->mlp_gate, ctx->activation->mlp_up);
            llaisysLinear(ctx->activation->mlp_down, ctx->activation->mlp_active, model->weights->mlp_down_w[i], NULL);
            llaisysAdd(ctx->activation->mlp_out, ctx->activation->mlp_residual, ctx->activation->mlp_down);
            tensorLoad(ctx->activation->attn_residual, ctx->activation->mlp_out->tensor->data());
        }
        llaisysRmsNorm(ctx->activation->out_norm, ctx->activation->mlp_out, model->weights->out_norm_w, model->meta->epsilon);
        llaisysLinear(ctx->activation->out_embed, ctx->activation->out_norm, model->weights->out_embed, NULL);
        llaisysTensor_t out_embed = tensorSlice(ctx->activation->out_embed, 0, ntoken-1, ntoken);
        llaisysArgmax(ctx->activation->max_token, ctx->activation->max_embed, out_embed);
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, LlaisysQwen2Context *ctx, int64_t * token_ids, size_t ntoken){
        //crete context seqlen = ntoken
        ctx->seqlen = ntoken;
        ctx->last_pos+= ntoken;
        int64_t first_token = llaisysQwen2ModelPrefill(model, ctx, token_ids, ntoken);
        //create context seqlen = 1 , save kvcache

        //decode seqlen=1

    }

}