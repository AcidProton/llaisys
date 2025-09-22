#include "llaisys_tensor.hpp"
#include "llaisys/models/qwen2.h"
#include "llaisys/ops.h"
#include "math.h"
#include <cstring>
#include <cmath>
#include <array>

__C{
    LlaisysQwen2Model *llaisysQwen2ModelCreate(LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int device_ids){
        // 在device上创建tensor，分配空间
        LlaisysQwen2Model *model = new LlaisysQwen2Model;
        model->meta = new LlaisysQwen2Meta;
        std::memcpy(model->meta, meta, sizeof(LlaisysQwen2Meta));
        model->device = device;
        model->device_ids = device_ids;
        model->weights = new LlaisysQwen2Weights;
        model->weights->in_embed = tensorCreate(std::array<size_t,2>{meta->voc,meta->hs}.data(),2,meta->dtype,device,device_ids);
        model->weights->out_embed = tensorCreate(std::array<size_t,2>{meta->voc,meta->hs}.data(),2,meta->dtype,device,device_ids);
        model->weights->out_norm_w = tensorCreate(std::array<size_t,1>{meta->hs}.data(),1,meta->dtype,device,device_ids);
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
            model->weights->attn_norm_w[i]=tensorCreate(std::array<size_t,1>{meta->hs}.data(),1,meta->dtype,device,device_ids);
            model->weights->attn_q_w[i]=tensorCreate(std::array<size_t,2>{meta->nh*meta->dh,meta->hs}.data(),2,meta->dtype,device,device_ids);
            model->weights->attn_q_b[i]=tensorCreate(std::array<size_t,1>{meta->nh*meta->dh}.data(),1,meta->dtype,device,device_ids);
            model->weights->attn_k_w[i]=tensorCreate(std::array<size_t,2>{meta->nkvh*meta->dh,meta->hs}.data(),2,meta->dtype,device,device_ids);
            model->weights->attn_k_b[i]=tensorCreate(std::array<size_t,1>{meta->nkvh*meta->dh}.data(),1,meta->dtype,device,device_ids);
            model->weights->attn_v_w[i]=tensorCreate(std::array<size_t,2>{meta->nkvh*meta->dh,meta->hs}.data(),2,meta->dtype,device,device_ids);
            model->weights->attn_v_b[i]=tensorCreate(std::array<size_t,1>{meta->nkvh*meta->dh}.data(),1,meta->dtype,device,device_ids);
            model->weights->attn_o_w[i]=tensorCreate(std::array<size_t,2>{meta->hs,meta->nh*meta->dh}.data(),2,meta->dtype,device,device_ids);
            model->weights->mlp_norm_w[i]=tensorCreate(std::array<size_t,1>{meta->hs}.data(),1,meta->dtype,device,device_ids);
            model->weights->mlp_up_w[i]=tensorCreate(std::array<size_t,2>{meta->di,meta->hs}.data(),2,meta->dtype,device,device_ids);
            model->weights->mlp_gate_w[i]=tensorCreate(std::array<size_t,2>{meta->di,meta->hs}.data(),2,meta->dtype,device,device_ids);
            model->weights->mlp_down_w[i]=tensorCreate(std::array<size_t,2>{meta->hs,meta->di}.data(),2,meta->dtype,device,device_ids);
        }
        
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

    void llaisysQwen2ModelDestroyContext(const LlaisysQwen2Meta *meta, LlaisysQwen2Context *context, bool destory_kvcache){
        tensorDestroy(context->activation->tokens);
        tensorDestroy(context->activation->pos_ids);
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
        tensorDestroy(context->activation->mlp_gate);
        tensorDestroy(context->activation->mlp_active);
        tensorDestroy(context->activation->mlp_up);
        tensorDestroy(context->activation->out_norm);
        tensorDestroy(context->activation->out_token_val);
        tensorDestroy(context->activation->max_token_val);
        tensorDestroy(context->activation->max_token_ids);
        if(destory_kvcache){
            for(size_t i=0;i<meta->nlayer;i++){
                tensorDestroy(context->kvcache->attn_v[i]);
                tensorDestroy(context->kvcache->attn_k_pos[i]);
            } 
        }
    }

    LlaisysQwen2Context *llaisysQwen2ModelUpdateContext(LlaisysQwen2Context *context, size_t seqlen){
        context->seqlen = seqlen;
        context->total_len += seqlen;
        int64_t* pos_ids = (int64_t*)tensorGetData(context->activation->pos_ids);
        for(size_t i=0;i<seqlen;i++){
            pos_ids[i] = (int64_t)(context->total_len- context->seqlen + i);
        }
        return context;
    }

    LlaisysQwen2Context *llaisysQwen2ModelCreateContext(LlaisysQwen2Model *model, LlaisysQwen2Context *last_context, size_t seqlen, size_t max_new_token){
        // naive，预先分配足够大的kvcache(total_len+max_new_token)
        LlaisysQwen2Context *context = new LlaisysQwen2Context;
        context->activation = new LlaisysQwen2Activation;
        context->kvcache = new LlaisysQwen2KVcache;
        llaisysDeviceType_t device = model->device;
        int device_ids = model->device_ids;
        LlaisysQwen2Meta *meta = model->meta;
        if(last_context==nullptr){ //for prefill
            context->seqlen = seqlen;
            context->total_len = seqlen;
            context->kvcache->attn_v = new llaisysTensor_t[meta->nlayer];
            for(size_t i=0;i<meta->nlayer;i++){
                context->kvcache->attn_v[i] = tensorCreate(std::array<size_t,3>{seqlen+max_new_token,meta->nkvh,meta->dh}.data(),3,meta->dtype,device,device_ids);
            }
            context->kvcache->attn_k_pos = new llaisysTensor_t[meta->nlayer];
            for (size_t i=0; i<meta->nlayer; i++){
                context->kvcache->attn_k_pos[i] = tensorCreate(std::array<size_t,3>{seqlen+max_new_token,meta->nkvh,meta->dh}.data(),3,meta->dtype,device,device_ids);
            }
        }else{ // for first decode
            context->seqlen = seqlen;
            context->total_len = last_context->total_len;
            context->kvcache->attn_v = last_context->kvcache->attn_v;
            context->kvcache->attn_k_pos = last_context->kvcache->attn_k_pos;
            llaisysQwen2ModelDestroyContext(meta, last_context, false);
        }
        context->activation->tokens = tensorCreate(std::array<size_t,1>{seqlen}.data(),1,LLAISYS_DTYPE_I64,device,device_ids);
        context->activation->pos_ids = tensorCreate(std::array<size_t,1>{seqlen}.data(),1,LLAISYS_DTYPE_I64,device,device_ids);
        int64_t* pos_ids = (int64_t*)tensorGetData(context->activation->pos_ids);
        for(size_t i=0;i<seqlen;i++){
            pos_ids[i] = (int64_t)(context->total_len - context->seqlen + i);
        }
        context->activation->in_embed = tensorCreate(std::array<size_t,2>{seqlen,meta->hs}.data(),2,meta->dtype,device,device_ids);
        context->activation->attn_residual = tensorCreate(std::array<size_t,2>{seqlen,meta->hs}.data(),2,meta->dtype,device,device_ids);
        context->activation->attn_norm = tensorCreate(std::array<size_t,2>{seqlen,meta->hs}.data(),2,meta->dtype,device,device_ids);
        context->activation->attn_q = tensorCreate(std::array<size_t,3>{seqlen,meta->nh,meta->dh}.data(),3,meta->dtype,device,device_ids);
        context->activation->attn_k = tensorCreate(std::array<size_t,3>{seqlen,meta->nkvh,meta->dh}.data(),3,meta->dtype,device,device_ids);
        context->activation->attn_q_pos = tensorCreate(std::array<size_t,3>{seqlen,meta->nh,meta->dh}.data(),3,meta->dtype,device,device_ids);
        context->activation->attn_val = tensorCreate(std::array<size_t,3>{seqlen,meta->nh,meta->dh}.data(),3,meta->dtype,device,device_ids);
        context->activation->attn_o = tensorCreate(std::array<size_t,2>{seqlen,meta->hs}.data(),2,meta->dtype,device,device_ids);
        context->activation->mlp_residual = tensorCreate(std::array<size_t,2>{seqlen,meta->hs}.data(),2,meta->dtype,device,device_ids);
        context->activation->mlp_norm = tensorCreate(std::array<size_t,2>{seqlen,meta->hs}.data(),2,meta->dtype,device,device_ids);
        context->activation->mlp_down = tensorCreate(std::array<size_t,2>{seqlen,meta->hs}.data(),2,meta->dtype,device,device_ids);
        context->activation->mlp_out = tensorCreate(std::array<size_t,2>{seqlen,meta->hs}.data(),2,meta->dtype,device,device_ids);
        context->activation->mlp_gate = tensorCreate(std::array<size_t,2>{seqlen,meta->di}.data(),2,meta->dtype,device,device_ids);
        context->activation->mlp_active = tensorCreate(std::array<size_t,2>{seqlen,meta->di}.data(),2,meta->dtype,device,device_ids);
        context->activation->mlp_up = tensorCreate(std::array<size_t,2>{seqlen,meta->di}.data(),2,meta->dtype,device,device_ids);
        context->activation->out_norm = tensorCreate(std::array<size_t,2>{size_t(1),meta->hs}.data(),2,meta->dtype,device,device_ids);
        context->activation->out_token_val = tensorCreate(std::array<size_t,2>{size_t(1),meta->voc}.data(),2,meta->dtype,device,device_ids);
        context->activation->max_token_val = tensorCreate(std::array<size_t,1>{size_t(1)}.data(),1,meta->dtype,device,device_ids);
        context->activation->max_token_ids = tensorCreate(std::array<size_t,1>{size_t(1)}.data(),1,llaisysDataType_t::LLAISYS_DTYPE_I64,device,device_ids);
        return context;
    } 

    int64_t llaisysQwen2ModelPrefill(LlaisysQwen2Model *model, LlaisysQwen2Context *ctx, int64_t *token_ids){
        //prefil seqlen=ntoken
        tensorLoad(ctx->activation->tokens, token_ids);
        llaisysEmbedding(ctx->activation->in_embed, ctx->activation->tokens, model->weights->in_embed);
        tensorLoad(ctx->activation->attn_residual, ctx->activation->in_embed->tensor->data());
        for(size_t i=0;i<model->meta->nlayer;i++){
            llaisysRmsNorm(ctx->activation->attn_norm, ctx->activation->attn_residual, model->weights->attn_norm_w[i], model->meta->epsilon);
            // qkv
            llaisysTensor_t attn_q_2d = tensorView(ctx->activation->attn_q,std::array<size_t,2>{ctx->seqlen*model->meta->nh,model->meta->dh}.data(),2);
            llaisysLinear(attn_q_2d, ctx->activation->attn_norm, model->weights->attn_q_w[i], model->weights->attn_q_b[i]);
            llaisysTensor_t attn_k_2d = tensorView(ctx->activation->attn_k,std::array<size_t,2>{ctx->seqlen*model->meta->nkvh,model->meta->dh}.data(),2);
            llaisysLinear(attn_k_2d, ctx->activation->attn_norm, model->weights->attn_k_w[i], model->weights->attn_k_b[i]);
            llaisysTensor_t attn_v=tensorSlice(ctx->kvcache->attn_v[i], 0, ctx->total_len-ctx->seqlen, ctx->total_len);
            llaisysTensor_t attn_v_2d = tensorView(attn_v,std::array<size_t,2>{ctx->seqlen*model->meta->nkvh,model->meta->dh}.data(),2);
            llaisysLinear(attn_v_2d, ctx->activation->attn_norm, model->weights->attn_v_w[i], model->weights->attn_v_b[i]);
            // q k rope
            llaisysROPE(ctx->activation->attn_q_pos, ctx->activation->attn_q, ctx->activation->pos_ids, model->meta->theta);
            llaisysTensor_t attn_k_pos=tensorSlice(ctx->kvcache->attn_k_pos[i], 0, ctx->total_len-ctx->seqlen, ctx->total_len);
            llaisysROPE(attn_k_pos, ctx->activation->attn_k, ctx->activation->pos_ids, model->meta->theta);
            //attn
            llaisysTensor_t attn_k_pos_total=tensorSlice(ctx->kvcache->attn_k_pos[i], 0, 0, ctx->total_len);
            llaisysTensor_t attn_v_total=tensorSlice(ctx->kvcache->attn_v[i], 0, 0, ctx->total_len);
            llaisysSelfAttention(ctx->activation->attn_val, ctx->activation->attn_q_pos, attn_k_pos_total, attn_v_total, (float)(1.0/sqrt(model->meta->dh)));
            llaisysTensor_t attn_val=tensorView(ctx->activation->attn_val,std::array<size_t,2>{ctx->seqlen,model->meta->nh*model->meta->dh}.data(),2);
            llaisysLinear(ctx->activation->attn_o, attn_val, model->weights->attn_o_w[i], nullptr);
            llaisysAdd(ctx->activation->mlp_residual, ctx->activation->attn_residual, ctx->activation->attn_o);
            //mlp
            llaisysRmsNorm(ctx->activation->mlp_norm, ctx->activation->mlp_residual, model->weights->mlp_norm_w[i], model->meta->epsilon);
            llaisysLinear(ctx->activation->mlp_gate, ctx->activation->mlp_norm, model->weights->mlp_gate_w[i], nullptr);
            llaisysLinear(ctx->activation->mlp_up, ctx->activation->mlp_norm, model->weights->mlp_up_w[i], nullptr);
            llaisysSwiGLU(ctx->activation->mlp_active, ctx->activation->mlp_gate, ctx->activation->mlp_up);
            llaisysLinear(ctx->activation->mlp_down, ctx->activation->mlp_active, model->weights->mlp_down_w[i], nullptr);
            llaisysAdd(ctx->activation->mlp_out, ctx->activation->mlp_residual, ctx->activation->mlp_down);
            tensorLoad(ctx->activation->attn_residual, ctx->activation->mlp_out->tensor->data());
        }
        llaisysTensor_t mlp_out = tensorSlice(ctx->activation->mlp_out, 0, ctx->seqlen-1, ctx->seqlen);
        llaisysRmsNorm(ctx->activation->out_norm, mlp_out, model->weights->out_norm_w, model->meta->epsilon);
        llaisysLinear(ctx->activation->out_token_val, ctx->activation->out_norm, model->weights->out_embed, nullptr);
        llaisysArgmax(ctx->activation->max_token_ids, ctx->activation->max_token_val, ctx->activation->out_token_val);
        int64_t* val_ptr = (int64_t*)ctx->activation->max_token_ids->tensor->data();
        return val_ptr[0];
    }

    void checkTensor(const char* tag, llaisysTensor_t t){
        bool toPrint = false;
        size_t numel = t->tensor->numel();
        llaisys::bf16_t* ptr = reinterpret_cast<llaisys::bf16_t *>(t->tensor->data());
        for(size_t i=0;i<numel;i++){
            llaisys::bf16_t* ele_ptr = ptr+i;
            float ele = llaisys::utils::cast<float>(*ele_ptr);
            if (std::isnan(ele) || std::abs(ele) > 1e6f) {
                printf("!discover illegal value: %s \n",tag);
                printf("total ele: %ld \n",numel);
                printf("illegal ele index: %ld \n",i);
                toPrint = true;
                break;
            }
        }//检查decode时，哪层的值变非法了
        if(toPrint){
            for(size_t i=0;i<numel;i++){
                llaisys::bf16_t* ele_ptr = ptr+i;
                float ele = llaisys::utils::cast<float>(*ele_ptr);
                printf("%f ",ele);
            }
            exit(9);
        }

    }

    int64_t llaisysQwen2ModelDecode(LlaisysQwen2Model *model, LlaisysQwen2Context *ctx, int64_t *token_ids){
        //prefil seqlen=ntoken
        tensorLoad(ctx->activation->tokens, token_ids);
        llaisysEmbedding(ctx->activation->in_embed, ctx->activation->tokens, model->weights->in_embed);
        tensorLoad(ctx->activation->attn_residual, ctx->activation->in_embed->tensor->data());
        for(size_t i=0;i<model->meta->nlayer;i++){
            // printf("in layer: %ld \n",i);
            llaisysRmsNorm(ctx->activation->attn_norm, ctx->activation->attn_residual, model->weights->attn_norm_w[i], model->meta->epsilon);
            // qkv
            llaisysTensor_t attn_q_2d = tensorView(ctx->activation->attn_q,std::array<size_t,2>{ctx->seqlen*model->meta->nh,model->meta->dh}.data(),2);
            llaisysLinear(attn_q_2d, ctx->activation->attn_norm, model->weights->attn_q_w[i], model->weights->attn_q_b[i]);
            llaisysTensor_t attn_k_2d = tensorView(ctx->activation->attn_k,std::array<size_t,2>{ctx->seqlen*model->meta->nkvh,model->meta->dh}.data(),2);
            llaisysLinear(attn_k_2d, ctx->activation->attn_norm, model->weights->attn_k_w[i], model->weights->attn_k_b[i]);
            llaisysTensor_t attn_v=tensorSlice(ctx->kvcache->attn_v[i], 0, ctx->total_len-ctx->seqlen, ctx->total_len);
            
            llaisysTensor_t attn_v_2d = tensorView(attn_v,std::array<size_t,2>{ctx->seqlen*model->meta->nkvh,model->meta->dh}.data(),2);
            llaisysLinear(attn_v_2d, ctx->activation->attn_norm, model->weights->attn_v_w[i], model->weights->attn_v_b[i]);
            //printf("debug attn_v\n");
            //tensorDebug(attn_v_2d);
            // q k rope
            llaisysROPE(ctx->activation->attn_q_pos, ctx->activation->attn_q, ctx->activation->pos_ids, model->meta->theta);
            llaisysTensor_t attn_k_pos=tensorSlice(ctx->kvcache->attn_k_pos[i], 0, ctx->total_len-ctx->seqlen, ctx->total_len);
            llaisysROPE(attn_k_pos, ctx->activation->attn_k, ctx->activation->pos_ids, model->meta->theta);
            //attn
            llaisysTensor_t attn_k_pos_total=tensorSlice(ctx->kvcache->attn_k_pos[i], 0, 0, ctx->total_len);
            llaisysTensor_t attn_v_total=tensorSlice(ctx->kvcache->attn_v[i], 0, 0, ctx->total_len);
            
            
            // checkTensor("attn_k_pos_total",attn_k_pos_total);
            // checkTensor("attn_v_total",attn_v_total);//v的cache可能越界
            llaisysSelfAttention(ctx->activation->attn_val, ctx->activation->attn_q_pos, attn_k_pos_total, attn_v_total, (float)(1.0/sqrt(model->meta->dh)));
            
            // checkTensor("ctx->activation->attn_val",ctx->activation->attn_val);
            llaisysTensor_t attn_val=tensorView(ctx->activation->attn_val,std::array<size_t,2>{ctx->seqlen,model->meta->nh*model->meta->dh}.data(),2);
            llaisysLinear(ctx->activation->attn_o, attn_val, model->weights->attn_o_w[i], nullptr);
            
            // checkTensor("ctx->activation->attn_o",ctx->activation->attn_o);
            llaisysAdd(ctx->activation->mlp_residual, ctx->activation->attn_residual, ctx->activation->attn_o);
            //mlp
            llaisysRmsNorm(ctx->activation->mlp_norm, ctx->activation->mlp_residual, model->weights->mlp_norm_w[i], model->meta->epsilon);
            llaisysLinear(ctx->activation->mlp_gate, ctx->activation->mlp_norm, model->weights->mlp_gate_w[i], nullptr);
            llaisysLinear(ctx->activation->mlp_up, ctx->activation->mlp_norm, model->weights->mlp_up_w[i], nullptr);
            llaisysSwiGLU(ctx->activation->mlp_active, ctx->activation->mlp_gate, ctx->activation->mlp_up);
            llaisysLinear(ctx->activation->mlp_down, ctx->activation->mlp_active, model->weights->mlp_down_w[i], nullptr);
            llaisysAdd(ctx->activation->mlp_out, ctx->activation->mlp_residual, ctx->activation->mlp_down);
            tensorLoad(ctx->activation->attn_residual, ctx->activation->mlp_out->tensor->data());
        }
        llaisysTensor_t mlp_out = tensorSlice(ctx->activation->mlp_out, 0, ctx->seqlen-1, ctx->seqlen);
        llaisysRmsNorm(ctx->activation->out_norm, mlp_out, model->weights->out_norm_w, model->meta->epsilon);
        llaisysLinear(ctx->activation->out_token_val, ctx->activation->out_norm, model->weights->out_embed, nullptr);
        llaisysArgmax(ctx->activation->max_token_ids, ctx->activation->max_token_val, ctx->activation->out_token_val);
        int64_t* val_ptr = (int64_t*)ctx->activation->max_token_ids->tensor->data();
        return val_ptr[0];
    }

    llaisysTensor_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t * token_ids, size_t ntoken, size_t max_new_token){
        printf("start infer\n");
        llaisysTensor_t out_token_tensor = tensorCreate(std::array<size_t,1>{max_new_token}.data(),1,LLAISYS_DTYPE_I64,model->device,model->device_ids);
        int64_t* out_token = (int64_t*)out_token_tensor->tensor->data();
        for(size_t i=0;i<max_new_token;i++){
            out_token[i] = (int64_t)(model->meta->end_token);
        }
        size_t token_cnt = 0;

        printf("prefill token:\n");
        for(size_t i=0;i<ntoken;i++){
            printf("%ld ",token_ids[i]);
        }

        //prefill seqlen = ntoken
        LlaisysQwen2Context* ctx = llaisysQwen2ModelCreateContext(model, nullptr, ntoken, max_new_token);
        int64_t cur_token = llaisysQwen2ModelPrefill(model, ctx, token_ids);
        out_token[token_cnt++] = cur_token;
        printf("finish prefill. get token: %ld\n",cur_token);

        //decode seqlen=1
        ctx = llaisysQwen2ModelCreateContext(model, ctx, 1, max_new_token);
        while(cur_token!=model->meta->end_token && token_cnt<max_new_token){
            ctx = llaisysQwen2ModelUpdateContext(ctx,1);
            cur_token = llaisysQwen2ModelDecode(model, ctx, &cur_token);
            out_token[token_cnt++] = cur_token;
            printf("decode one round. get token: %ld\n",cur_token);
        }
        llaisysQwen2ModelDestroyContext(model->meta, ctx, true);
        return out_token_tensor;
    }

} 