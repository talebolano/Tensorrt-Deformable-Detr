#pragma once
#include <assert.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <unordered_map>
#include <NvInfer.h>
#include <NvInferVersion.h>

#define DEVICE 0
#define BATCH_SIZE 1
using namespace nvinfer1;

#if NV_TENSORRT_MAJOR > 7
#define PLUGIN_NOEXCEPT noexcept 
#define MatrixKNONE MatrixOperation::kNONE
#define MatrixTRANS MatrixOperation::kTRANSPOSE
#else
#define PLUGIN_NOEXCEPT
#define MatrixKNONE false
#define MatrixTRANS true
#endif
static const float SCALING = 0.17677669529663687;
static const float SCALING_ONE = 1.0;
static const float SHIFT_ZERO = 0.0;
static const float POWER_TWO = 2.0;
static const float EPS = 0.00001;
static const float ZEROFIVE = 0.5;



ITensor* PositionEmbeddingSine(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
ITensor& input,  // B,C,H,W
int num_pos_feats = 64,
int temperature = 10000
) {
    // out 1,H*W,C
    // refer to https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py#12
    // TODO: improve this implementation
    auto mask_dim = input.getDimensions(); // B,C,H,W
    int h = mask_dim.d[2], w = mask_dim.d[3];
    std::vector<std::vector<float>> y_embed(h);
    for (int i = 0; i < h; i++)
        y_embed[i] = std::vector<float>(w, i + 1);
    std::vector<float> sub_embed(w, 0);
    for (int i = 0; i < w; i++)
        sub_embed[i] = i + 1;
    std::vector<std::vector<float>> x_embed(h, sub_embed);

    // normalize
    float eps = 1e-6, scale = 2.0 * 3.1415926;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            y_embed[i][j] = y_embed[i][j] / (h + eps) * scale;
            x_embed[i][j] = x_embed[i][j] / (w + eps) * scale;
        }
    }

    // dim_t
    std::vector<float> dim_t(num_pos_feats, 0);
    for (int i = 0; i < num_pos_feats; i++) {
        dim_t[i] = pow(temperature, (2 * (i / 2) / static_cast<float>(num_pos_feats)));
    }

    // pos_x, pos_y
    std::vector<std::vector<std::vector<float>>> pos_x(h,
    std::vector<std::vector<float>>(w,
    std::vector<float>(num_pos_feats, 0)));

    std::vector<std::vector<std::vector<float>>> pos_y(h,
    std::vector<std::vector<float>>(w,
    std::vector<float>(num_pos_feats, 0)));
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < num_pos_feats; k++) {
                float value_x = x_embed[i][j] / dim_t[k];
                float value_y = y_embed[i][j] / dim_t[k];
                if (k & 1) {
                    pos_x[i][j][k] = std::cos(value_x);
                    pos_y[i][j][k] = std::cos(value_y);
                } else {
                    pos_x[i][j][k] = std::sin(value_x);
                    pos_y[i][j][k] = std::sin(value_y);
                }
            }
        }
    }

    // pos
    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * h * w * num_pos_feats * 2));
    float *pNext = pval;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int k = 0; k < num_pos_feats; k++) {
                *pNext = pos_y[i][j][k];
                ++pNext;
            }
            for (int k = 0; k < num_pos_feats; k++) {
                *pNext = pos_x[i][j][k];
                ++pNext;
            }
        }
    }
    Weights pos_embed_weight{ DataType::kFLOAT, pval, h * w * num_pos_feats * 2 };
    auto pos_embed = network->addConstant(Dims3{ 1,h * w, num_pos_feats * 2 }, pos_embed_weight);
    assert(pos_embed);
    return pos_embed->getOutput(0);
}




ITensor* MultiHeadAttention(
INetworkDefinition *network,
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
ITensor& query, // nq,bs,c
ITensor& key,   // nq,bs,c
ITensor& value,
int embed_dim = 256,
int num_heads = 8
) {
    int batch_size = query.getDimensions().d[1];
    int tgt_len = query.getDimensions().d[0];  // nq, bs, c
    int head_dim = embed_dim / num_heads;

    // q
    auto q_weights = network->addConstant(Dims3{1,embed_dim,embed_dim},weightMap[lname + ".in_proj_weight_q"]);
    auto linear_q = network->addMatrixMultiply(
        query,
        MatrixKNONE,
        *q_weights->getOutput(0),
        MatrixKNONE);
    assert(linear_q);
    auto q_bais = network->addConstant(Dims3{1,1,embed_dim},weightMap[lname + ".in_proj_bias_q"]);
    auto linear_q_bias = network->addElementWise(
        *linear_q->getOutput(0),
        *q_bais->getOutput(0),
        ElementWiseOperation::kSUM
    );


    // k
    auto k_weights = network->addConstant(Dims3{1,embed_dim,embed_dim},weightMap[lname + ".in_proj_weight_k"]);
    auto linear_k = network->addMatrixMultiply(
        key,
        MatrixKNONE,
        *k_weights->getOutput(0),
        MatrixKNONE);
    assert(linear_k);
    auto k_bais = network->addConstant(Dims3{1,1,embed_dim},weightMap[lname + ".in_proj_bias_k"]);
    auto linear_k_bias = network->addElementWise(
        *linear_k->getOutput(0),
        *k_bais->getOutput(0),
        ElementWiseOperation::kSUM
    );

    // v
    auto v_weights = network->addConstant(Dims3{1,embed_dim,embed_dim},weightMap[lname + ".in_proj_weight_v"]);
    auto linear_v = network->addMatrixMultiply(
        value,
        MatrixKNONE,
        *v_weights->getOutput(0),
        MatrixKNONE);
    assert(linear_v);
    auto v_bais = network->addConstant(Dims3{1,1,embed_dim},weightMap[lname + ".in_proj_bias_v"]);
    auto linear_v_bias = network->addElementWise(
        *linear_v->getOutput(0),
        *v_bais->getOutput(0),
        ElementWiseOperation::kSUM
    );

    auto scaling_t = network->addConstant(Dims3{ 1, 1, 1}, Weights{ DataType::kFLOAT, &SCALING, 1 });
    assert(scaling_t);
    auto q_scaling = network->addElementWise(
        *linear_q_bias->getOutput(0),
        *scaling_t->getOutput(0),
        ElementWiseOperation::kPROD); //q,bs,c
    assert(q_scaling);

    auto q_shuffle = network->addShuffle(*q_scaling->getOutput(0));
    assert(q_shuffle);
    q_shuffle->setName((lname + ".q_shuffle").c_str());
    q_shuffle->setReshapeDimensions(Dims3{ tgt_len, batch_size*num_heads, head_dim });  // q,bs,c --> q,bs*nh,hd
    q_shuffle->setSecondTranspose(Permutation{1, 0, 2}); // bs*n_h, q, c

    auto k_shuffle = network->addShuffle(*linear_k_bias->getOutput(0));
    assert(k_shuffle);
    k_shuffle->setName((lname + ".k_shuffle").c_str());
    k_shuffle->setReshapeDimensions(Dims3{ tgt_len, batch_size*num_heads, head_dim });
    k_shuffle->setSecondTranspose(Permutation{ 1, 0, 2 }); // bs*n_h, q, c

    auto v_shuffle = network->addShuffle(*linear_v_bias->getOutput(0));
    assert(v_shuffle);
    v_shuffle->setName((lname + ".v_shuffle").c_str());
    v_shuffle->setReshapeDimensions(Dims3{ tgt_len, batch_size*num_heads, head_dim });
    v_shuffle->setSecondTranspose(Permutation{ 1, 0, 2 });

    auto q_product_k = network->addMatrixMultiply(*q_shuffle->getOutput(0), MatrixKNONE, *k_shuffle->getOutput(0), MatrixTRANS);
    assert(q_product_k); //n_h*bs,q,  q

    // src_key_padding_mask are all false, so do nothing here
    // see https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/nn/functional/activation.h#826-#839

    auto softmax = network->addSoftMax(*q_product_k->getOutput(0));
    assert(softmax);
    softmax->setAxes(2); //

    auto attn_product_v = network->addMatrixMultiply(*softmax->getOutput(0), MatrixKNONE, *v_shuffle->getOutput(0), MatrixKNONE);
    assert(attn_product_v); // nh*bs,q,c

    auto attn_shuffle = network->addShuffle(*attn_product_v->getOutput(0));
    assert(attn_shuffle);
    attn_shuffle->setName((lname + ".attn_shuffle").c_str());
    attn_shuffle->setFirstTranspose(Permutation{ 1, 0, 2 }); // q,bs*nh,c
    attn_shuffle->setReshapeDimensions(Dims3{ tgt_len, batch_size, embed_dim }); // q,bs,c


    auto linear_out_weights = network->addConstant(Dims3{1,embed_dim,embed_dim},weightMap[lname + ".out_proj.weight"]);
    auto linear_attn = network->addMatrixMultiply(
        *attn_shuffle->getOutput(0),
        MatrixKNONE,
        *linear_out_weights->getOutput(0),
        MatrixKNONE);
    assert(linear_attn);
    auto linear_out_bias = network->addConstant(Dims3{1,1,embed_dim},weightMap[lname + ".out_proj.bias"]);
    auto linear_attn_bias = network->addElementWise(
        *linear_attn->getOutput(0),
        *linear_out_bias->getOutput(0),
        ElementWiseOperation::kSUM
    );
    assert(linear_attn_bias);

    return linear_attn_bias->getOutput(0);
}


ITensor* LayerNorm(
INetworkDefinition *network,
ITensor& input, // q,bs,c
std::unordered_map<std::string, Weights>& weightMap,
const std::string& lname,
int d_model = 256
) {
    // TODO: maybe a better implementation https://github.com/NVIDIA/TensorRT/blob/master/plugin/common/common.cuh#212
    auto mean = network->addReduce(input, ReduceOperation::kAVG, 2, true);
    assert(mean); //q,bs,1

    auto sub_mean = network->addElementWise(input, *mean->getOutput(0), ElementWiseOperation::kSUB);
    assert(sub_mean);// q,bs,c

    // implement pow2 with scale
    auto pow_power = network->addConstant(Dims3{ 1, 1, 1}, Weights{ DataType::kFLOAT, &POWER_TWO, 1 });
    auto pow2 = network->addElementWise(*sub_mean->getOutput(0),*pow_power->getOutput(0),ElementWiseOperation::kPOW);
    assert(pow2);

    auto pow_mean = network->addReduce(*pow2->getOutput(0), ReduceOperation::kAVG, 2, true);
    assert(pow_mean);

    auto eps = network->addConstant(Dims3{ 1, 1, 1 }, Weights{ DataType::kFLOAT, &EPS, 1 });
    assert(eps);

    auto add_eps = network->addElementWise(*pow_mean->getOutput(0), *eps->getOutput(0), ElementWiseOperation::kSUM);
    assert(add_eps);

    auto sqrt = network->addUnary(*add_eps->getOutput(0), UnaryOperation::kSQRT);
    assert(sqrt);

    auto div = network->addElementWise(*sub_mean->getOutput(0), *sqrt->getOutput(0), ElementWiseOperation::kDIV);
    assert(div);

    auto affine_weight = network->addConstant(Dims3{1,1,d_model}, weightMap[lname + ".weight"]);
    auto affine_multi = network->addElementWise(*div->getOutput(0),
                                    *affine_weight->getOutput(0),
                                    ElementWiseOperation::kPROD);
    auto affine_bias = network->addConstant(Dims3{1,1,d_model},weightMap[lname + ".bias"]);
    auto affine_add = network->addElementWise(*affine_multi->getOutput(0),
                                    *affine_bias->getOutput(0),
                                    ElementWiseOperation::kSUM);

    assert(affine_add);
    return affine_add->getOutput(0);
}


ITensor* FFN(
    INetworkDefinition *network,
    ITensor& input, // q,bs,c
    std::unordered_map<std::string, Weights>& weightMap,
    const std::string& lname,
    int d_model = 256,
    int feedward_ratio=4
){

    auto linear1_weights = network->addConstant(Dims3{1,d_model,feedward_ratio*d_model},weightMap[lname + ".layers.0.0.weight"]);
    auto linear1_mul = network->addMatrixMultiply(
        input,
        MatrixKNONE,
        *linear1_weights->getOutput(0),
        MatrixKNONE);
    assert(linear1_mul);
    auto linear1_bias = network->addConstant(Dims3{1,1,feedward_ratio*d_model},weightMap[lname + ".layers.0.0.bias"]);
    auto linear1_add = network->addElementWise(
        *linear1_mul->getOutput(0),
        *linear1_bias->getOutput(0),
        ElementWiseOperation::kSUM
    );
    assert(linear1_add);

    auto activation = network->addActivation(*linear1_add->getOutput(0),ActivationType::kRELU);

    auto linear2_weights = network->addConstant(Dims3{1,feedward_ratio*d_model,d_model},weightMap[lname + ".layers.1.weight"]);
    auto linear2_mul = network->addMatrixMultiply(
        *activation->getOutput(0),
        MatrixKNONE,
        *linear2_weights->getOutput(0),
        MatrixKNONE);
    assert(linear2_mul);
    auto linear2_bias = network->addConstant(Dims3{1,1,d_model},weightMap[lname + ".layers.1.bias"]);
    auto linear2_add = network->addElementWise(
        *linear2_mul->getOutput(0),
        *linear2_bias->getOutput(0),
        ElementWiseOperation::kSUM
    );
    assert(linear2_add);
    return linear2_add->getOutput(0);
}


//TODO: 添加 kaypadding mask
ITensor* MultiScaleDeformHeadAttention(
    INetworkDefinition *network,
    std::unordered_map<std::string, Weights>& weightMap,
    const std::string& lname,
    ITensor& query, // nq,bs,c
    ITensor& value, // hw1+hw2+hw3+hw4,bs,c
    ITensor& reference_points, // bs, num_query, num_levels, 2 or bs, num_query, num_levels, 4
    ITensor& spatial_shapes, // num_level,2 hw
    ITensor& level_start_index, // (num_levels, )
    int embed_dim = 256,
    int num_heads = 8,
    int num_level = 4,
    int num_point = 4
){

    int batch_size = query.getDimensions().d[1];
    int tgt_len = query.getDimensions().d[0];
    int num_value = value.getDimensions().d[1];

    auto q_stuffle = network->addShuffle(query);
    q_stuffle->setName((lname+".q_shuffle").c_str());
    q_stuffle->setFirstTranspose(Permutation{1,0,2});

    auto v_stuffle = network->addShuffle(value);
    v_stuffle->setName((lname+".v_shuffle").c_str());
    v_stuffle->setFirstTranspose(Permutation{1,0,2});

    auto value_proj_weights = network->addConstant(Dims3{1,embed_dim,embed_dim},weightMap[lname + ".value_proj.weight"]);
    auto value_proj_mul = network->addMatrixMultiply(
        *v_stuffle->getOutput(0),
        MatrixKNONE,
        *value_proj_weights->getOutput(0),
        MatrixKNONE);
    assert(value_proj_mul);
    auto value_proj_bais = network->addConstant(Dims3{1,1,embed_dim},weightMap[lname + ".value_proj.bias"]);
    auto value_proj_add = network->addElementWise(
        *value_proj_mul->getOutput(0),
        *value_proj_bais->getOutput(0),
        ElementWiseOperation::kSUM
    );
    assert(value_proj_add);

    // TODO: 在value添加keypadding mask
    auto v_stuffle_2 = network->addShuffle(*value_proj_add->getOutput(0));
    v_stuffle->setName((lname+".v_shuffle_2").c_str());
    v_stuffle->setReshapeDimensions(Dims4{batch_size,num_value,num_heads,-1});    

    // sampling_offset
    auto sampling_offset_weights = network->addConstant(
                    Dims3{1,num_heads * num_level * num_point * 2,num_heads * num_level * num_point * 2},
                    weightMap[lname+".sampling_offsets.weight"]);
    auto sampling_offset_mul = network->addMatrixMultiply(
        *q_stuffle->getOutput(0),
        MatrixKNONE,
        *sampling_offset_weights->getOutput(0),
        MatrixKNONE
    );
    assert(sampling_offset_mul);
    auto sampling_offset_bias = network->addConstant(Dims3{1,1,num_heads * num_level * num_point * 2},
                    weightMap[lname+".sampling_offsets.bias"]);
    auto sampling_offset_add = network->addElementWise(
        *sampling_offset_mul->getOutput(0),
        *sampling_offset_bias->getOutput(0),
        ElementWiseOperation::kSUM
    );
    assert(sampling_offset_add);

    auto sampling_offset_shuffle = network->addShuffle(*sampling_offset_add->getOutput(0));
    sampling_offset_shuffle->setName((lname+".sampling_shuffle").c_str());
    sampling_offset_shuffle->setReshapeDimensions(Dims{6, {batch_size, tgt_len, num_heads, num_level, num_point,2}});
    assert(sampling_offset_shuffle);

    // attention_weights
    auto attn_weight_weights = network->addConstant(
                    Dims3{1,num_heads * num_level * num_point,num_heads * num_level * num_point},
                    weightMap[lname+".attention_weights.weight"]);
    auto attn_weight_mul = network->addMatrixMultiply(
        *q_stuffle->getOutput(0),
        MatrixKNONE,
        *attn_weight_weights->getOutput(0),
        MatrixKNONE
    );
    assert(attn_weight_mul);
    auto attn_weight_bias = network->addConstant(
                    Dims3{1,1,num_heads * num_level * num_point},
                    weightMap[lname+".attention_weights.bias"]);
    auto attn_weight_add = network->addElementWise(
        *attn_weight_mul->getOutput(0),
        *attn_weight_bias->getOutput(0),
        ElementWiseOperation::kSUM
    );
    assert(attn_weight_add); 

    auto attn_weight_shuffle = network->addShuffle(*attn_weight_add->getOutput(0));
    attn_weight_shuffle->setName((lname+".attn_weight_shuffle").c_str());
    attn_weight_shuffle->setReshapeDimensions(Dims4{batch_size, tgt_len, num_heads, num_level*num_point});
    assert(attn_weight_shuffle);       

    auto attn_weight_softmax = network->addSoftMax(*attn_weight_shuffle->getOutput(0));
    attn_weight_softmax->setAxes(3);
    assert(attn_weight_softmax);

    auto attn_weight_shuffle2 = network->addShuffle(*attn_weight_softmax->getOutput(0));
    attn_weight_shuffle2->setName((lname+".attn_weight_shuffle2").c_str());
    attn_weight_shuffle2->setReshapeDimensions(Dims{5,{batch_size, tgt_len, num_heads, num_level,num_point}});
    assert(attn_weight_shuffle2);

    IElementWiseLayer* sampling_locations;
    // refernce point norm   
    int point_size = reference_points.getDimensions().d[3];  
    if (point_size==2){
            // offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)  
            //TODO: 检查是否正确
            auto spatial_shapes_W = network->addSlice(spatial_shapes,Dims2{0,1},Dims2{num_level,1},Dims2{1,1}); //num_level,1
            auto spatial_shapes_H = network->addSlice(spatial_shapes,Dims2{0,0},Dims2{num_level,1},Dims2{1,1}); //num_level,1
            ITensor * spatial_shapes_WH[] = {spatial_shapes_W->getOutput(0),spatial_shapes_H->getOutput(0)};
            auto offset_normalizer = network->addConcatenation(spatial_shapes_WH,2);
            offset_normalizer->setAxis(1);

            auto reference_point_shuffle = network->addShuffle(reference_points);
            reference_point_shuffle->setName((lname+".reference_point_shuffle").c_str());
            reference_point_shuffle->setReshapeDimensions(Dims{6,{batch_size,tgt_len,1,num_level,1,point_size}});

            auto offset_normalizer_shuffle = network->addShuffle(*offset_normalizer->getOutput(0));
            offset_normalizer_shuffle->setName((lname+".offset_normalizer_shuffle").c_str());
            offset_normalizer_shuffle->setReshapeDimensions(Dims{6,{1,1,1,num_level,1,point_size}});

            auto sampling_offset_norm = network->addElementWise(
                *sampling_offset_shuffle->getOutput(0),
                *offset_normalizer_shuffle->getOutput(0),
                ElementWiseOperation::kDIV
            );

            auto sampling_locations = network->addElementWise(
                *reference_point_shuffle->getOutput(0),
                *sampling_offset_norm->getOutput(0),
                ElementWiseOperation::kSUM
            );
    }    
    else if (point_size==4){
            auto reference_point_shuffle = network->addShuffle(reference_points);
            reference_point_shuffle->setName((lname+".reference_point_shuffle").c_str());
            reference_point_shuffle->setReshapeDimensions(Dims{6,{batch_size,tgt_len,1,num_level,1,point_size}});

            auto reference_point_cxcy = network->addSlice(
                                *reference_point_shuffle->getOutput(0),
                                Dims{6,{0,0,0,0,0,0}},
                                Dims{6,{batch_size,tgt_len,1,num_level,1,2}},
                                Dims{6,{1,1,1,1,1,1}});

            auto reference_point_wh = network->addSlice(
                                *reference_point_shuffle->getOutput(0),
                                Dims{6,{0,0,0,0,0,2}},
                                Dims{6,{batch_size,tgt_len,1,num_level,1,2}},
                                Dims{6,{1,1,1,1,1,1}});

            
            auto sampling_offset_num_points = network->addConstant(Dims{6,{1,1,1,1,1,1}},Weights{DataType::kFLOAT, &num_point, 1});
            auto sampling_offset_norm = network->addElementWise(
                *sampling_offset_shuffle->getOutput(0),
                *sampling_offset_num_points->getOutput(0),
                ElementWiseOperation::kDIV
            );

            auto sampling_offset_norm2 = network->addElementWise(
                *sampling_offset_norm->getOutput(0),
                *reference_point_wh->getOutput(0),
                ElementWiseOperation::kPROD
            );


            auto sampling_offset_five = network->addConstant(Dims{6,{1,1,1,1,1,1}},Weights{DataType::kFLOAT, &ZEROFIVE, 1});
            auto sampling_offset_norm3 = network->addElementWise(
                *sampling_offset_norm2->getOutput(0),
                *sampling_offset_five->getOutput(0),
                ElementWiseOperation::kPROD
            );

            auto sampling_locations = network->addElementWise(
                *reference_point_cxcy->getOutput(0),
                *sampling_offset_norm3->getOutput(0),
                ElementWiseOperation::kSUM
            );            
    }
    
    auto creator = getPluginRegistry()->getPluginCreator("MultiscaleDeformableAttnPlugin_TRT","1");
    const PluginFieldCollection* plugin_data = creator->getFieldNames();
    IPluginV2* plugin_obj = creator->createPlugin((lname+".MultiscaleDeformableAttn").c_str(),plugin_data);

    ITensor * mshda[] = {v_stuffle->getOutput(0),&spatial_shapes,&level_start_index,sampling_locations->getOutput(0),attn_weight_shuffle2->getOutput(0)};
    //do sample plugin
    auto multi_scale_deformable_attn = network->addPluginV2(mshda,5,*plugin_obj);

    auto multi_scale_deformable_attn_shuffle = network->addShuffle(*multi_scale_deformable_attn->getOutput(0));
    multi_scale_deformable_attn_shuffle->setName((lname+".multi_scale_deformable_attn_shuffle").c_str());
    multi_scale_deformable_attn_shuffle->setReshapeDimensions(Dims3(batch_size,tgt_len,embed_dim));
    multi_scale_deformable_attn_shuffle->setSecondTranspose(Permutation{1,0,2}); // num_q,bs,c

    auto output_proj_weight = network->addConstant(Dims3{1,embed_dim,embed_dim},weightMap[lname+".output_proj.weight"]);
    auto output_proj_mul = network->addMatrixMultiply(
        *multi_scale_deformable_attn_shuffle->getOutput(0),
        MatrixKNONE,
        *output_proj_weight->getOutput(0),
        MatrixKNONE
    );

    auto output_proj_bias = network->addConstant(Dims3{1,1,embed_dim},weightMap[lname+".output_proj.bias"]);

    auto output_proj_add = network->addElementWise(
        *output_proj_mul->getOutput(0),
        *output_proj_bias->getOutput(0),
        ElementWiseOperation::kSUM
    );
    return output_proj_add->getOutput(0);
}


ITensor* TransformerEncoderLayer(
    INetworkDefinition *network,
    std::unordered_map<std::string, Weights>& weightMap,
    const std::string& lname,
    ITensor& query, //  hw1+hw2+hw3+hw4,bs,c
    ITensor& query_embed,// hw1+hw2+hw3+hw4,bs,c
    ITensor& reference_points, // bs, num_query, num_levels, 2 
    ITensor& spatial_shapes, // num_level,2 hw
    ITensor& level_start_index // (num_levels, )
){

    auto query_embed_add = network->addElementWise(
                            query,
                            query_embed,
                            ElementWiseOperation::kSUM);

    ITensor* query2 = MultiScaleDeformHeadAttention(network,weightMap,lname+".attentions.0",
                                    *query_embed_add->getOutput(0),
                                    query,
                                    reference_points,
                                    spatial_shapes,
                                    level_start_index,256,8,4,4);
    
    auto Identity1 = network->addElementWise(
        *query2,
        query,
        ElementWiseOperation::kSUM
    );
    
    ITensor* norm1 = LayerNorm(network,*Identity1->getOutput(0),weightMap,lname+".norms.0",256);

    ITensor* ffn1 = FFN(network,*norm1,weightMap,lname+".ffn.0",256,4);

    auto Identity2 = network->addElementWise(
        *ffn1,
        *norm1,
        ElementWiseOperation::kSUM
    );

    ITensor* norm2 = LayerNorm(network,*Identity2->getOutput(0),weightMap,lname+".norms.1",256);

    return norm2;
}

ITensor* TransformerEncoder(
    INetworkDefinition *network,
    std::unordered_map<std::string, Weights>& weightMap,
    const std::string& lname,
    ITensor& query, //  hw1+hw2+hw3+hw4,bs,c
    ITensor& query_embed,// hw1+hw2+hw3+hw4,bs,c
    ITensor& reference_points, // bs, num_query, num_levels, 2 
    ITensor& spatial_shapes, // num_level,2 hw
    ITensor& level_start_index, // (num_levels, )
    int num_layers = 6
){
    ITensor* out = &query;
    for (int i = 0; i < num_layers; i++) {
        std::string layer_name = lname + ".layers." + std::to_string(i);
        out = TransformerEncoderLayer(network, weightMap, layer_name, *out, 
                    query_embed,reference_points,spatial_shapes,level_start_index);
    }
    return out;    
}

ITensor* TransformerDecoderLayer(
    INetworkDefinition *network,
    std::unordered_map<std::string, Weights>& weightMap,
    const std::string& lname,
    ITensor& query, //  nq,bs,c
    ITensor& query_embed,// nq,bs,c
    ITensor& value, // hw1+hw2+hw3+hw4,bs,c
    ITensor& reference_points, // bs, num_query, 4 
    ITensor& spatial_shapes, // num_level,2 hw
    ITensor& level_start_index // (num_levels, )
){

    //reference_points bs, num_query, 4 -> bs, num_query,4, 4
    int batch_size = reference_points.getDimensions().d[0];
    int tgt_len = reference_points.getDimensions().d[1];

    auto reference_points_num_level = network->addShuffle(reference_points);
    reference_points_num_level->setName((lname+".reference_points_num_level").c_str());

    auto valid_ratios = network->addConstant(Dims4{1,1,4,1},Weights{DataType::kFLOAT,&SCALING_ONE,1});
    reference_points_num_level->setReshapeDimensions(Dims4{batch_size,tgt_len,1,4});
    auto reference_points_num_level2 = network->addElementWise(
        *reference_points_num_level->getOutput(0),
        *valid_ratios->getOutput(0),
        ElementWiseOperation::kPROD
    );


    auto query_embed_add1 = network->addElementWise(
                            query,
                            query_embed,
                            ElementWiseOperation::kSUM);

    ITensor* query1 = MultiHeadAttention(network,weightMap,lname+".attentions.0.attn",
                                    *query_embed_add1->getOutput(0),
                                    *query_embed_add1->getOutput(0),
                                    query,256,8);

    auto Identity1 = network->addElementWise(
        *query1,
        query,
        ElementWiseOperation::kSUM
    );

    ITensor* norm1 = LayerNorm(network,*Identity1->getOutput(0),weightMap,lname+".norms.0",256);

    auto query_embed_add2 = network->addElementWise(
                            *norm1,
                            query_embed,
                            ElementWiseOperation::kSUM);
                            
    ITensor* query2 = MultiScaleDeformHeadAttention(network,weightMap,lname+".attentions.1",
                                    *query_embed_add2->getOutput(0),
                                    value,
                                    *reference_points_num_level2->getOutput(0),
                                    spatial_shapes,
                                    level_start_index,256,8,4,4);

    auto Identity2= network->addElementWise(
        *query2,
        *norm1,
        ElementWiseOperation::kSUM
    );

    ITensor* norm2 = LayerNorm(network,*Identity2->getOutput(0),weightMap,lname+".norms.1",256);

    ITensor* ffn1 = FFN(network,*norm2,weightMap,lname+".ffn.0",256,4);

    auto Identity3= network->addElementWise(
        *ffn1,
        *norm2,
        ElementWiseOperation::kSUM
    );

    ITensor* norm3 = LayerNorm(network,*Identity3->getOutput(0),weightMap,lname+".norms.2",256);

    return norm3;
}

ITensor* ClsRegBranch(
    INetworkDefinition *network,
    std::unordered_map<std::string, Weights>& weightMap,
    const std::string& lname,
    ITensor& input, //  bs, num_query, 4
    int d_model = 256,
    int out_dim = 4
){

    auto linear1_weights = network->addConstant(Dims3{1,d_model,d_model},weightMap[lname + ".0.weight"]);
    auto linear1_mul = network->addMatrixMultiply(
        input,
        MatrixKNONE,
        *linear1_weights->getOutput(0),
        MatrixKNONE);
    assert(linear1_mul);
    auto linear1_bias = network->addConstant(Dims3{1,1,d_model},weightMap[lname + ".0.bias"]);
    auto linear1_add = network->addElementWise(
        *linear1_mul->getOutput(0),
        *linear1_bias->getOutput(0),
        ElementWiseOperation::kSUM
    );
    assert(linear1_add);

    auto activation1 = network->addActivation(*linear1_add->getOutput(0),ActivationType::kRELU);

    auto linear2_weights = network->addConstant(Dims3{1,d_model,d_model},weightMap[lname + ".1.weight"]);
    auto linear2_mul = network->addMatrixMultiply(
        *activation1->getOutput(0),
        MatrixKNONE,
        *linear2_weights->getOutput(0),
        MatrixKNONE);
    assert(linear2_mul);
    auto linear2_bias = network->addConstant(Dims3{1,1,d_model},weightMap[lname + ".1.bias"]);
    auto linear2_add = network->addElementWise(
        *linear2_mul->getOutput(0),
        *linear2_bias->getOutput(0),
        ElementWiseOperation::kSUM
    );
    assert(linear2_add);

    auto activation2 = network->addActivation(*linear2_add->getOutput(0),ActivationType::kRELU);

    auto linear3_weights = network->addConstant(Dims3{1,d_model,out_dim},weightMap[lname + ".2.weight"]);
    auto linear3_mul = network->addMatrixMultiply(
        *activation2->getOutput(0),
        MatrixKNONE,
        *linear3_weights->getOutput(0),
        MatrixKNONE);
    assert(linear3_mul);
    auto linear3_bias = network->addConstant(Dims3{1,1,out_dim},weightMap[lname + ".2.bias"]);
    auto linear3_add = network->addElementWise(
        *linear3_mul->getOutput(0),
        *linear3_bias->getOutput(0),
        ElementWiseOperation::kSUM
    );
    assert(linear3_add);     

    return linear1_add->getOutput(0);
}


ITensor* InverseSigmoid(
    INetworkDefinition *network,
    const std::string& lname,    
    ITensor& input //  bs,nq,4   
){

    auto clamp_max = network->addConstant(Dims3{1,1,1},Weights{DataType::kFLOAT,&SCALING_ONE,1});
    auto clamp_min = network->addConstant(Dims3{1,1,1},Weights{DataType::kFLOAT,&EPS,1});

    auto clamp1 = network->addElementWise(
        input,
        *clamp_max->getOutput(0),
        ElementWiseOperation::kMIN
    );

    auto clamp2 = network->addElementWise(
        *clamp1->getOutput(0),
        *clamp_min->getOutput(0),
        ElementWiseOperation::kMAX
    );    

    auto one_sub_x = network->addElementWise(
        *clamp_max->getOutput(0),
        *clamp2->getOutput(0),
        ElementWiseOperation::kSUB
    );

    auto clamp3 = network->addElementWise(
        *one_sub_x->getOutput(0),
        *clamp_min->getOutput(0),
        ElementWiseOperation::kMAX
    );      

    auto x1_div_x2 = network->addElementWise(
        *clamp2->getOutput(0),
        *clamp3->getOutput(0),
        ElementWiseOperation::kDIV
    );

    auto log_x = network->addUnary(
        *x1_div_x2->getOutput(0),
        UnaryOperation::kLOG

    );

    return log_x->getOutput(0);
}


ITensor** TransformerDecoder(
    INetworkDefinition *network,
    std::unordered_map<std::string, Weights>& weightMap,
    const std::string& lname,
    ITensor& query, //  nq,bs,c
    ITensor& query_embed,// nq,bs,c
    ITensor& value, // hw1+hw2+hw3+hw4,bs,c
    ITensor& reference_points, // bs, num_query, 4
    ITensor& spatial_shapes, // num_level,2 hw
    ITensor& level_start_index, // (num_levels, )
    int num_layers = 6
){
    ITensor* out = &query;
    ITensor* reference_points_update = &reference_points;

    for (int i = 0; i < num_layers; i++) {
        std::string layer_name = lname + ".layers." + std::to_string(i);
        out = TransformerDecoderLayer(network, weightMap, layer_name, *out, 
                    query_embed,value,*reference_points_update,spatial_shapes,level_start_index);

        // get bbox and update
        auto out_shuffle1 = network->addShuffle(*out);
        out_shuffle1->setName((lname+".out_shuffle1."+std::to_string(i)).c_str());
        out_shuffle1->setFirstTranspose(Permutation{1, 0, 2 });

        ITensor* reference_points_delta = ClsRegBranch(network,weightMap,
                                            "bbox_head.reg_branches."+std::to_string(i),
                                            *out_shuffle1->getOutput(0),256,4);

        ITensor* reference_points_inverse_sigmoid = InverseSigmoid(network,lname+".inverse_sigmoid",*reference_points_update);

        auto new_refernce_point_inverse_sigmoid = network->addElementWise(
                                            *reference_points_inverse_sigmoid,
                                            *reference_points_delta,
                                            ElementWiseOperation::kSUM);
        
        auto new_refernce_point = network->addUnary(
                                            *new_refernce_point_inverse_sigmoid->getOutput(0),
                                            UnaryOperation::kEXP);
        reference_points_update = new_refernce_point->getOutput(0);
    }

    ITensor * decoder_output[] = {out,reference_points_update};
    return decoder_output;    
}


ITensor* EncoderReferencePoints(
    INetworkDefinition *network,
    const std::string& lname,
    std::vector<std::vector<int>> spatial_shapes // num_level,2
){
    // 返回固定值的reference point shape (bs, num_keys, num_levels, 2).
    int reference_points_size = 0;
    for(int i=0;i<spatial_shapes.size();++i){
        auto spatial_shape = spatial_shapes[i];    
        int h = spatial_shape[0];
        int w = spatial_shape[1];

        reference_points_size+=h*w;

    }

    float *reference_points = reinterpret_cast<float*>(malloc(sizeof(float) * reference_points_size * 2));// hxw+hxw+hxw,2

    for(int i=0;i<spatial_shapes.size();++i){
        auto spatial_shape = spatial_shapes[i];    
        int h = spatial_shape[0];
        int w = spatial_shape[1];

        for(int y=0;y<h;++y){
            for(int x=0;x<w;++x){
                reference_points[2*x + 2*y*w + 2*i*h*w] = (0.5 + x)/(float)w;
                reference_points[2*x + 1 + 2*y*w + 2*i*h*w] = (0.5 + y)/(float)h;
            }
        }

    }

    Weights reference_points_weight{DataType::kFLOAT,reference_points,reference_points_size * 2};

    auto reference_points_constant = network->addConstant(
                    Dims4{1,reference_points_size,1,2},reference_points_weight);


    auto valid_ratios = network->addConstant(Dims4{1,1,spatial_shapes.size(),1},Weights{DataType::kFLOAT,&SCALING_ONE,1});

    auto reference_points_num_level = network->addElementWise(
        *reference_points_constant->getOutput(0),
        *valid_ratios->getOutput(0),
        ElementWiseOperation::kPROD
    );

    return reference_points_num_level->getOutput(0);
}


ITensor** GenEncoderOutputProposals(
    INetworkDefinition *network,
    std::unordered_map<std::string, Weights>& weightMap,
    const std::string& lname,
    ITensor &memory, //(bs, num_key, embed_dim) 
    std::vector<std::vector<int>> spatial_shapes,
    int d_model = 256
){
    // 返回固定值的reference point shape (bs, num_keys, 4).
    int reference_points_size = 0;
    for(int i=0;i<spatial_shapes.size();++i){
        auto spatial_shape = spatial_shapes[i];    
        int h = spatial_shape[0];
        int w = spatial_shape[1];

        reference_points_size+=h*w;

    }

    float *reference_points = reinterpret_cast<float*>(malloc(sizeof(float) * reference_points_size * 2));// hxw+hxw+hxw,4

    for(int i=0;i<spatial_shapes.size();++i){
        auto spatial_shape = spatial_shapes[i];    
        int h = spatial_shape[0];
        int w = spatial_shape[1];

        for(int y=0;y<h;++y){
            for(int x=0;x<w;++x){
                reference_points[4*x + 4*y*w + 4*i*h*w] = (0.5 + x)/(float)w;
                reference_points[4*x + 1 + 4*y*w + 4*i*h*w] = (0.5 + y)/(float)h;
                reference_points[4*x + 2 + 4*y*w + 4*i*h*w] = 0.05*pow(2.,(float)i);
                reference_points[4*x + 3 + 4*y*w + 4*i*h*w] = 0.05*pow(2.,(float)i);
            }
        }

    }

    Weights reference_points_weight{DataType::kFLOAT,reference_points,reference_points_size * 4};   

    auto output_proposal = network->addConstant(
                    Dims3{1,reference_points_size,4},reference_points_weight);

    ITensor* output_proposal_inverse_sigmoid = InverseSigmoid(network,lname+".inverse_sigmoid",*output_proposal->getOutput(0));

    // output memory
    auto enc_output_weight = network->addConstant(Dims3{1,d_model,d_model},
                            weightMap[lname+".weight"]);

    auto enc_output_mul = network->addMatrixMultiply(
                            memory,
                            MatrixKNONE,
                            *enc_output_weight->getOutput(0),
                            MatrixKNONE
                            );

    auto enc_output_bias = network->addConstant(Dims3{1,1,d_model},weightMap[lname+",bias"]);

    auto enc_output_add = network->addElementWise(
                            *enc_output_mul->getOutput(0),
                            *enc_output_bias->getOutput(0),
                            ElementWiseOperation::kSUM
                        );

    ITensor* outputs[] = {enc_output_add->getOutput(0),output_proposal_inverse_sigmoid};

    return outputs;
}


ITensor* ProPosalPosEmbed(
    INetworkDefinition* network,
    const std::string& lname,
    ITensor& output_proposal,  // B,nq,4 sigmoid
    int num_pos_feats = 128,
    int temperature = 10000
){

    int batch_size = output_proposal.getDimensions().d[0];
    int tgt_len = output_proposal.getDimensions().d[1];

    float *dim_t = reinterpret_cast<float*>(malloc(sizeof(float) * num_pos_feats));// 128

    for(int i=0;i<num_pos_feats;++i){
        dim_t[i] = 2.*3.1415926 / pow((float)temperature, (float)(2.*(i/2) / (float)num_pos_feats));
    }

    auto dim = network->addConstant(Dims4{1,1,1,num_pos_feats},Weights{DataType::kFLOAT,&dim_t,num_pos_feats});

    auto proposals_shuffle = network->addShuffle(output_proposal);
    proposals_shuffle->setName((lname+".shuffle").c_str());
    proposals_shuffle->setReshapeDimensions(Dims4{batch_size,tgt_len,4,1});

    auto pos = network->addElementWise(
                    *proposals_shuffle->getOutput(0),
                    *dim->getOutput(0),
                    ElementWiseOperation::kPROD); //b,nq,4,128

    auto pos_cxw = network->addSlice(
                    *pos->getOutput(0),
                    Dims4{0,0,0,0},
                    Dims4{batch_size,tgt_len,4,num_pos_feats/2},
                    Dims4{1,1,1,2}); //b,nq,4,64

    auto pos_cyh = network->addSlice(
                    *pos->getOutput(0),
                    Dims4{0,0,0,1},
                    Dims4{batch_size,tgt_len,4,num_pos_feats/2},
                    Dims4{1,1,1,2}); //b,nq,4,64

    auto pos_cxw_sin = network->addUnary(
                    *pos_cxw->getOutput(0),
                    UnaryOperation::kSIN);

    auto pos_cxw_cos = network->addUnary(
                    *pos_cyh->getOutput(0),
                    UnaryOperation::kCOS);

    ITensor * pos_cxwcyh[] = {pos_cxw_sin->getOutput(0),pos_cxw_cos->getOutput(0)};
    auto pos_cat = network->addConcatenation(
                        pos_cxwcyh,
                        2);
    pos_cat->setAxis(3);
    auto pos_shuffle = network->addShuffle(*pos_cat->getOutput(0));
    pos_shuffle->setName((lname+".pos_shuffle").c_str());
    pos_shuffle->setReshapeDimensions(Dims3{batch_size,tgt_len,4*num_pos_feats});

    return pos_shuffle->getOutput(0);

}

ITensor** Transformer(
    INetworkDefinition *network,
    std::unordered_map<std::string, Weights>& weightMap,
    const std::string& lname,
    ITensor& mlvl_feats // 
){
    
}






