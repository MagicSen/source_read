#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

// Yolo层
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;

    // n: 实际使用的anchor box数目 3, total: 总anchor box数目 9, h, w 都是上层传下来的参数, 应该是grid_h, grid_w
    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    // yolo层输出: channels 数目等于 anchor box数目 * (类别数 + x, y, w, h + confidence)
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total*2, sizeof(float));
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            // 未输入mask, 则初始化 mask为 0,1,2,3...8
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    // 每个图像最多识别90个框
    l.truths = 90*(4 + 1);
    // 创建内部缓存
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        // biases初始化为0.5, 之后自动填充为预设的anchor
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

void resize_yolo_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =     cuda_make_array(l->delta, l->batch*l->outputs);
    l->output_gpu =    cuda_make_array(l->output, l->batch*l->outputs);
#endif
}

// 根据网络输入层数据以及位置坐标，得到矩形框，矩形框: [x, y, w, h]
// 输入：
//      x：      上层网络
//      biases:  anchor box的尺寸，对应取第6,7,8个anchor
//      n：      确定取第几个anchor的下标
//      index:   输入层网络下标
//      i：      输入层x下标
//      j：      输入层y下标
//      lw：     输入层width
//      lh：     输入层height
//      w：      图像的width
//      h：      图像的height
//      stride： 输入层一个channel的步长: lw*lh
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    // 4个channel, 1对应x, 2对应y, 3对应width, 4对应height
    // 论文中:
    //          x = delta(tx) + cx
    //          y = delta(ty) + cy
    //          w = pw * exp(tw)
    //          h = ph * exp(th)
    // 其中：
    //          cx = i
    //          cy = j
    //          tx = x[index + 0 * stride]
    //          ty = x[index + 1 * stride]
    //          pw = biases[2*n]            // 预先设置的anchor_box的width与height
    //          ph = biases[2*n + 1]
    //          tw = x[index + 2 * stride]
    //          th = y[index + 3 * stride]
    // 此处除以lw, lh, w, h为框的归一化操作，为了与ground_truth直接比较
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

// 计算框的残差
// 输入：
//      truth：  真实框
//      x：      预测值
//      biases： anchor box的尺寸
//      n：      采用anchor box的下标
//      index：  预测层下标
//      i：      grid_width 下标
//      j：      grid_height 下标
//      lw：     grid_width
//      lh：     grid_height
//      w：      图像宽
//      h：      图像高
//      delta：  残差指针
//      scale：  学习尺度 = 2 - 真实框的w * h 
//      stride： 步长
float delta_yolo_box(box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);
    // 反向计算求差
    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

// 计算类别误差
// 输入：
//      delta：  误差项
//      index：  预测类别的下标
//      class：  框的类别
//      classes：总类别数
//      stride： 类别步长
//      avg_cat：0
void delta_yolo_class(float *output, float *delta, int index, int class, int classes, int stride, float *avg_cat)
{
    int n;
    // 如果第一个类别刚好为正确样本，直接计算loss，返回
    if (delta[index]){
        // 设置confidence为 1 - 第n个类的概率
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    for(n = 0; n < classes; ++n){
        // 设置其他类别的残差，对于其他类别，这个类为背景
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
}

// l为该层的引用，batch为第几波数据，location为具体位置，entry为具体nchannel的偏于量
static int entry_index(layer l, int batch, int location, int entry)
{
    // n表示第几个nchannel, loc表示位于这个channel的具体位置
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    // 返回输出层的下标
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
}


// Yolo正向传递
void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    // 输入拷贝至输出
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
#endif

    // 初始化delta为全0
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    // 遍历上层输入
    for (b = 0; b < l.batch; ++b) {
        // 修改残差的值
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                // 此处的n为anchor的数目，这里表示有3个anchor
                for (n = 0; n < l.n; ++n) {
                    // 得到grid_w,grid_h中某个节点对应回归的矩形框
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    // Step 1: 找到该anchor box最匹配的类
                    // 比较选择iou最大的矩形框, 每个图像最多识别90个框，因此ground_truth对应最多有90个，实际有很多为0
                    for(t = 0; t < l.max_boxes; ++t){
                        // 得到该grid_w,grid_h中某个节点下对应的坐标框
                        // 第一个参数：第一项net.truth为label的头指针，第二项为一个图像内部框的偏移量，第三项为第几个图像
                        // 第二个参数：channel的步长
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        // 一个图像内不可能都有max个矩形框，这里提前退出
                        if(!truth.x) break;
                        // 计算预测框与真实框的IOU
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    // Step 2：计算该anchor box最佳匹配的confidence
                    // 第5个nchannels为检测框类别的confidence
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    // 默认delta为 -l.output[obj_index]
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    // 默认ignore_thresh = 0.5
                    if (best_iou > l.ignore_thresh) {
                        // 如果iou满足ignore_thresh，忽略
                        l.delta[obj_index] = 0;
                    }
                    // Step 3：如果best_iou满足阈值，设置class loss以及回归框loss
                    // 计算类别误差, 默认 truth_thresh = 1.0
                    if (best_iou > l.truth_thresh) {
                        // 如果best_iou满足truth_thresh，则delta为 1 - l.output[obj_index] 
                        l.delta[obj_index] = 1 - l.output[obj_index];
                        // 得到这个框对应的类别
                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        // 得到预测类别的下标
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        // 得到iou最大的类别框
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }
        // 为什么还需要匹配一次呢？
        // 上面是基于预测结果循环得到每个anchor box对应的最大iou
        //  每幅图像真实检测框 + 匹配最大IOU的anchor box
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            // 得到一个图像中的最大IOU矩形框
            float best_iou = 0;
            int best_n = 0;
            // 真实值相对grid_width，grid_height的位置
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            // 适配其他未采用的anchor box，匹配最优iou
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }
            // 如果最佳匹配的mask在所选的anchor box中，更新残差矩阵
            // 残差矩阵二次匹配，修正delta的值
            int mask_n = int_index(l.mask, best_n, l.n);
            if(mask_n >= 0){
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                float iou = delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, net.w, net.h, l.delta, (2-truth.w*truth.h), l.w*l.h);

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = 1 - l.output[obj_index];

                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    // sum(diff*diff)
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

void backward_yolo_layer(const layer l, network net)
{
   // 将yolo计算的残差留给下层
   axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}

int yolo_num_detections(layer l, float thresh)
{
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if(l.output[obj_index] > thresh){
                ++count;
            }
        }
    }
    return count;
}

void avg_flipped_yolo(layer l)
{
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for(z = 0; z < l.classes + 4 + 1; ++z){
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if(z == 0){
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                    }
                }
            }
        }
    }
    for(i = 0; i < l.outputs; ++i){
        l.output[i] = (l.output[i] + flip[i])/2.;
    }
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets)
{
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if(objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for(j = 0; j < l.classes; ++j){
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net)
{
    copy_gpu(l.batch*l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    int b, n;
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array_gpu(l.output_gpu + index, 2*l.w*l.h, LOGISTIC);
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array_gpu(l.output_gpu + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    }
    if(!net.train || l.onlyforward){
        cuda_pull_array(l.output_gpu, l.output, l.batch*l.outputs);
        return;
    }

    cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
    forward_yolo_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
}

void backward_yolo_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

