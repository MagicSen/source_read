#pragma once
#ifndef CAFFE_REG_H
#define CAFFE_REG_H


//#define CPU_ONLY 0
#include "caffe/common.hpp"  
#include "caffe/layers/input_layer.hpp"  
#include "caffe/layers/inner_product_layer.hpp"  
#include "caffe/layers/dropout_layer.hpp"  
#include "caffe/layers/conv_layer.hpp"  
#include "caffe/layers/relu_layer.hpp"  

#include "caffe/layers/pooling_layer.hpp"  
#include "caffe/layers/lrn_layer.hpp"  
#include "caffe/layers/softmax_layer.hpp"  
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/layers/concat_yolo_layer.hpp"
#include "caffe/layers/yolo_v3_loss_layer.hpp"
#include "caffe/layers/upsample_layer.hpp"
#include "caffe/layers/mvn_layer.hpp"
namespace caffe
{
	extern INSTANTIATE_CLASS(InputLayer);
	//extern INSTANTIATE_CLASS(ConvolutionLayer);
	//REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);

	//extern INSTANTIATE_CLASS(ReLULayer);
	//REGISTER_LAYER_CLASS(ReLU);
	//extern INSTANTIATE_CLASS(PoolingLayer);
	//REGISTER_LAYER_CLASS(Pooling);
	//extern INSTANTIATE_CLASS(LRNLayer);
	//REGISTER_LAYER_CLASS(LRN);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	//REGISTER_LAYER_CLASS(Normalize);
	extern INSTANTIATE_CLASS(FlattenLayer);
	extern INSTANTIATE_CLASS(ConcatLayer);
	extern INSTANTIATE_CLASS(ReshapeLayer);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	//REGISTER_LAYER_CLASS(Softmax);
	extern INSTANTIATE_CLASS(BatchNormLayer);
	extern INSTANTIATE_CLASS(ScaleLayer);
	extern INSTANTIATE_CLASS(BiasLayer);
	extern INSTANTIATE_CLASS(ConcatYoloLayer);
	extern INSTANTIATE_CLASS(YoloV3LossLayer);
	extern INSTANTIATE_CLASS(UpsampleLayer);
	extern INSTANTIATE_CLASS(MVNLayer);
}

#endif
