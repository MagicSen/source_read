/**
 * @brief A layer factory that allows one to register layers.
 * During runtime, registered layers can be called by passing a LayerParameter
 * protobuffer to the CreateLayer function:
 *
 *     LayerRegistry<Dtype>::CreateLayer(param);
 *
 * There are two ways to register a layer. Assuming that we have a layer like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeLayer : public Layer<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Layer" at the end
 * ("MyAwesomeLayer" -> "MyAwesome").
 *
 * If the layer is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_LAYER_CLASS(MyAwesome);
 *
 * Or, if the layer is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Layer<Dtype*> GetMyAwesomeLayer(const LayerParameter& param) {
 *      // your implementation
 *    }
 *
 * (for example, when your layer has multiple backends, see GetConvolutionLayer
 * for a use case), then you can register the creator function instead, like
 *
 * REGISTER_LAYER_CREATOR(MyAwesome, GetMyAwesomeLayer)
 *
 * Note that each layer type should only be registered once.
 */

#ifndef CAFFE_LAYER_FACTORY_H_
#define CAFFE_LAYER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Layer;

template <typename Dtype>
class LayerRegistry {
 public:
  // 定义Creator函数指针，接收LayerParameter为参数，返回shared_ptr类型的Layer共享指针
  typedef shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
  typedef std::map<string, Creator> CreatorRegistry;

  // 静态方法，只能在本文件内部使用，返回一个静态map用来记录layername与layer的映射关系
  // 静态数据成员(只存在一份)，生命周期伴随整个程序，使用范围仅限该类内部
  static CreatorRegistry& Registry() {
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    // 返回查找key对应的元素个数，如果为1表示已经在map中，注册过
    CHECK_EQ(registry.count(type), 0)
        << "Layer type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a layer using a LayerParameter.
  static shared_ptr<Layer<Dtype> > CreateLayer(const LayerParameter& param) {
    if (Caffe::root_solver()) {
      LOG(INFO) << "Creating layer " << param.name();
    }
    const string& type = param.type();
    // 得到全局map，里面存放已经注册好的Layer创建方法，这里需要基于LayerParameter参数初始化
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 1) << "Unknown layer type: " << type
        << " (known types: " << LayerTypeListString() << ")";
    return registry[type](param);
  }

  static vector<string> LayerTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> layer_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      layer_types.push_back(iter->first);
    }
    return layer_types;
  }

 private:
  // Layer registry should never be instantiated - everything is done with its
  // 注册层不能实例化，他只能是静态变量
  // static variables.
  LayerRegistry() {}

  // 这个函数只负责将整个可用layer名称输出
  static string LayerTypeListString() {
    vector<string> layer_types = LayerTypeList();
    string layer_types_str;
    for (vector<string>::iterator iter = layer_types.begin();
         iter != layer_types.end(); ++iter) {
      if (iter != layer_types.begin()) {
        layer_types_str += ", ";
      }
      layer_types_str += *iter;
    }
    return layer_types_str;
  }
};


template <typename Dtype>
class LayerRegisterer {
 public:
  // 构造函数
  LayerRegisterer(const string& type,
                  shared_ptr<Layer<Dtype> > (*creator)(const LayerParameter&)) {
    // LOG(INFO) << "Registering layer type: " << type;
    LayerRegistry<Dtype>::AddCreator(type, creator);
  }
};

// 产生全局网络层的构造函数，输入Layer名称以及Layer指针，将该类注册到全局变量中
// C++宏定义中，##表示宏连接，#表示字符串行化，例如 type为 Convolution, #type 表示 “Convolution”, A_##type表示 A_Convolution
#define REGISTER_LAYER_CREATOR(type, creator)                                  \
  static LayerRegisterer<float> g_creator_f_##type(#type, creator<float>);     \
  static LayerRegisterer<double> g_creator_d_##type(#type, creator<double>)    \

#define REGISTER_LAYER_CLASS(type)                                             \
  template <typename Dtype>                                                    \
  shared_ptr<Layer<Dtype> > Creator_##type##Layer(const LayerParameter& param) \
  {                                                                            \
    return shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));           \
  }                                                                            \
  REGISTER_LAYER_CREATOR(type, Creator_##type##Layer)

}  // namespace caffe

#endif  // CAFFE_LAYER_FACTORY_H_
