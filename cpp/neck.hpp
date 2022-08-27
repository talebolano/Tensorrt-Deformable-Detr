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
static const float SCALING_ONE = 1.0;
static const float SHIFT_ZERO = 0.0;
static const float POWER_TWO = 2.0;
static const float EPS = 0.00001;
static const float ZEROFIVE = 0.5;


ITensor* GroupNorm(
    INetworkDefinition *network,
    std::unordered_map<std::string, Weights>& weightMap,
    const std::string& lname,
    ITensor &input, // b,c,h,w
    int num_group=32
){
    int batch_size = input.getDimensions().d[0];
    int channel = input.getDimensions().d[1];
    int H = input.getDimensions().d[2];
    int W = input.getDimensions().d[3];

    auto group_shuffle = network->addShuffle(input);
    group_shuffle->setName((lname+".group_shuffle").c_str());
    group_shuffle->setReshapeDimensions(Dims3{batch_size,num_group,H*W*channel/num_group});// bs,num_g,c

    auto mean = network->addReduce(*group_shuffle->getOutput(0), ReduceOperation::kAVG, 2, true);
    assert(mean); //bs,num_g,1

    auto sub_mean = network->addElementWise(*group_shuffle->getOutput(0), *mean->getOutput(0), ElementWiseOperation::kSUB);
    assert(sub_mean);//bs,num_g,1

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

    auto group_shuffle2 = network->addShuffle(*div->getOutput(0));
    group_shuffle2->setName((lname+".group_shuffle2").c_str());
    group_shuffle2->setReshapeDimensions(Dims4{batch_size,channel,H,W});// bs,c,h,w

    auto affine_weight = network->addConstant(Dims4{1,channel,1,1}, weightMap[lname + ".weight"]);
    auto affine_multi = network->addElementWise(*group_shuffle2->getOutput(0),
                                    *affine_weight->getOutput(0),
                                    ElementWiseOperation::kPROD);

    auto affine_bias = network->addConstant(Dims4{1,channel,1,1},weightMap[lname + ".bias"]);
    auto affine_add = network->addElementWise(*affine_multi->getOutput(0),
                                    *affine_bias->getOutput(0),
                                    ElementWiseOperation::kSUM);
    assert(affine_add);
    return affine_add->getOutput(0);
}


std::vector<ITensor*> ChannelMapper(
    std::vector<ITensor* > &inputs,
    INetworkDefinition* network,
    std::unordered_map<std::string, Weights>& weightMap,
    const std::string& lname,
    int out_channels=256
){
    // conv1
    auto conv1 = network->addConvolutionNd(
        *inputs[0],
        out_channels,
        DimsHW{ 1, 1 },
        weightMap[lname + ".convs.0.conv.weight"],
        Weights{});
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 1, 1 });
    conv1->setPaddingNd(DimsHW{ 0, 0 });

    ITensor* output1 = GroupNorm(network,weightMap,lname+".convs.0.gn",*conv1->getOutput(0),32);

    // conv2
    auto conv2 = network->addConvolutionNd(
        *inputs[1],
        out_channels,
        DimsHW{ 1, 1 },
        weightMap[lname + ".convs.1.conv.weight"],
        Weights{});
    assert(conv2);
    conv2->setStrideNd(DimsHW{ 1, 1 });
    conv2->setPaddingNd(DimsHW{ 0, 0 });

    ITensor* output2 = GroupNorm(network,weightMap,lname+".convs.1.gn",*conv2->getOutput(0),32);

    // conv3
    auto conv3 = network->addConvolutionNd(
        *inputs[2],
        out_channels,
        DimsHW{ 1, 1 },
        weightMap[lname + ".convs.2.conv.weight"],
        Weights{});
    assert(conv3);
    conv3->setStrideNd(DimsHW{ 1, 1 });
    conv3->setPaddingNd(DimsHW{ 0, 0 });

    ITensor* output3 = GroupNorm(network,weightMap,lname+".convs.2.gn",*conv3->getOutput(0),32);

    // conv4
    auto conv4 = network->addConvolutionNd(
        *inputs[2],
        out_channels,
        DimsHW{ 3, 3 },
        weightMap[lname + ".extra_convs.0.conv.weight"],
        Weights{});
    assert(conv3);
    conv4->setStrideNd(DimsHW{ 2, 2 });
    conv4->setPaddingNd(DimsHW{ 1, 1 });

    ITensor* output4 = GroupNorm(network,weightMap,lname+".extra_convs.0.gn",*conv4->getOutput(0),32);    

    std::vector<ITensor*>  neck_output = {output1,output2,output3,output4};

    return neck_output;
}



