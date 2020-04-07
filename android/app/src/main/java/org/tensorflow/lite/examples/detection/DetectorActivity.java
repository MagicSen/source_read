/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
// 入口Activity，继承自CameraActivity，实现Camera 2.0的 图像获取回调函数(需要实现onImageAvailable接口)
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  // 开启日志
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
//  private static final int TF_OD_API_INPUT_SIZE = 300;
//  private static final boolean TF_OD_API_IS_QUANTIZED = true;
//  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
//  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
/*
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final int TF_OD_API_INPUT_HEIGHT = 240;
  private static final int TF_OD_API_INPUT_WIDTH = 320;

 private static final boolean TF_OD_API_IS_QUANTIZED = false;
 private static final String TF_OD_API_MODEL_FILE = "face_detect.tflite";

  //private static final boolean TF_OD_API_IS_QUANTIZED = true;
  //private static final String TF_OD_API_MODEL_FILE = "face_detect_quan.tflite";
*/

/*
  private static final int TF_OD_API_INPUT_HEIGHT = 240;
  private static final int TF_OD_API_INPUT_WIDTH = 320;

  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "face_detect_20191129.tflite";
  //private static final boolean TF_OD_API_IS_QUANTIZED = true;
  //private static final String TF_OD_API_MODEL_FILE = "face_detect_quantized_20191202.tflite";

  //private static final boolean TF_OD_API_IS_QUANTIZED = true;
  //private static final String TF_OD_API_MODEL_FILE = "face_detect_quan.tflite";

  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/facelabelmap.txt";
*/

  // 设置tensorflow模型输入尺寸
  private static final int TF_OD_API_INPUT_HEIGHT = 180;
  private static final int TF_OD_API_INPUT_WIDTH = 320;

//  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  //private static final String TF_OD_API_MODEL_FILE = "hand_detect_20191230.tflite";
//  private static final String TF_OD_API_MODEL_FILE = "hand_detect_20200326.tflite";

  // 设置模型路径以及是否开启量化
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "hand_detect_quantized_20200401.tflite";

  //private static final boolean TF_OD_API_IS_QUANTIZED = true;
  //private static final String TF_OD_API_MODEL_FILE = "face_detect_quantized_20191202.tflite";

  //private static final boolean TF_OD_API_IS_QUANTIZED = true;
  //private static final String TF_OD_API_MODEL_FILE = "face_detect_quan.tflite";

  // 设置label文件路径
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/handlabelmap.txt";

  // 设置检测模型类型，这里是tf模型
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // 设置检测框置信度
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  // 从相机获得的图像，变换到模型输入图像是否保持尺寸
  private static final boolean MAINTAIN_ASPECT = true;
  // 相机预览图片尺寸
  //private static final Size DESIRED_PREVIEW_SIZE = new Size(320, 240);
  private static final Size DESIRED_PREVIEW_SIZE = new Size(320, 180);

  // 预览结果是否保存本地
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  // 文本框文本dip分辨率
  private static final float TEXT_SIZE_DIP = 10;
  // view上的浮层
  OverlayView trackingOverlay;
  // 获取传感器方向
  private Integer sensorOrientation;

  // 检测器
  private Classifier detector;
  // 上一帧处理时长
  private long lastProcessingTimeMs;
  // 图像变量
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  // 检测是异步进行的，这里判断是否启用检测
  private boolean computingDetection = false;

  private long timestamp = 0;

  // 从相机图像 ==> 模型输入图像 (尺寸放缩，传感器位置转换)
  private Matrix frameToCropTransform;
  // 模型输入图像 ==> 相机图像 逆变换
  private Matrix cropToFrameTransform;

  // 跟踪框
  private MultiBoxTracker tracker;

  // 绘制文本工具
  private BorderedText borderedText;

/*
  // resize图像，弃用
  public Bitmap resizeImage(Bitmap bitmap, int w, int h) {
    Bitmap BitmapOrg = bitmap;
    int width = BitmapOrg.getWidth();
    int height = BitmapOrg.getHeight();
    int newWidth = w;
    int newHeight = h;

    float scaleWidth = ((float) newWidth) / width;
    float scaleHeight = ((float) newHeight) / height;

    Matrix matrix = new Matrix();
    matrix.postScale(scaleWidth, scaleHeight);
    // if you want to rotate the Bitmap
    // matrix.postRotate(45);
    Bitmap resizedBitmap = Bitmap.createBitmap(BitmapOrg, 0, 0, width,
            height, matrix, true);
    return resizedBitmap;
  }

  // 弃用
  public Bitmap getResizedBitmap(Bitmap bm, int newWidth, int newHeight) {
    int width = bm.getWidth();
    int height = bm.getHeight();
    float scaleWidth = ((float) newWidth) / width;
    float scaleHeight = ((float) newHeight) / height;
    // CREATE A MATRIX FOR THE MANIPULATION
    Matrix matrix = new Matrix();
    // RESIZE THE BIT MAP
    matrix.postScale(scaleWidth, scaleHeight);

    // "RECREATE" THE NEW BITMAP
    Bitmap resizedBitmap = Bitmap.createBitmap(
            bm, 0, 0, width, height, matrix, false);
    bm.recycle();
    return resizedBitmap;
  }
*/
  // 第一次启动，初始化缓存，变换矩阵，设置浮层绘图回调函数
  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    //int cropSize = TF_OD_API_INPUT_SIZE;

    int cropHeight = TF_OD_API_INPUT_HEIGHT;
    int cropWidth = TF_OD_API_INPUT_WIDTH;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              //TF_OD_API_INPUT_SIZE,
                  TF_OD_API_INPUT_HEIGHT,
                  TF_OD_API_INPUT_WIDTH,
              TF_OD_API_IS_QUANTIZED);
      //cropSize = TF_OD_API_INPUT_SIZE;
      cropHeight = TF_OD_API_INPUT_HEIGHT;
      cropWidth = TF_OD_API_INPUT_WIDTH;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation() + 90;
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    //croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            //cropSize, cropSize,
                cropWidth, cropHeight,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    // 获得浮层
    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    // 设置回调函数
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  // 获取图像，执行预测
  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    // 刷新浮层结果
    trackingOverlay.postInvalidate();

    // 仍然检测的时候，刷新图像缓存，获取下一帧
    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    // debug: 方便截图
    // load bitmap from path
    //String img_path="/storage/emulated/0/tensorflow/bbb.png";
    //rgbFrameBitmap = BitmapFactory.decodeFile(img_path);
    // 得到当前帧结果，参数pixels, offset, stride, x, y, width, height
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    //String img_saved_path="input.png";
    //ImageUtils.saveBitmap(rgbFrameBitmap, img_saved_path);
    // 获取下一帧
    readyForNextImage();

    // 得到模型的输入图像，其中frameToCropTransform为 相机图片到模型输入图片的变换矩阵
    final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    //canvas.drawBitmap(rgbFrameBitmap,  new Matrix(), null);
    // For examining the actual TF input.
    //croppedBitmap = resizeImage(croppedBitmap, 320, 240);
    //croppedBitmap = getResizedBitmap(croppedBitmap, 320, 240);
    /*
    String img_saved_path_src="input.png";
    ImageUtils.saveBitmap(rgbFrameBitmap, img_saved_path_src);
    String img_saved_path="crop.png";
    ImageUtils.saveBitmap(croppedBitmap, img_saved_path);
    */
    // 保存结果
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    // 设置异步预测进程
    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            // 得到当前时间戳
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            // 预测
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            //String img_saved_path="crop_input.png";
            //ImageUtils.saveBitmap(croppedBitmap, img_saved_path);
            // 获取预测耗时
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            // 重新拷贝canvas
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);
            // 设置检测阈值
            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            // LinkedList属于双向链表
            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            // 处理结果
            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              // 大于阈值的，导入结果list
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);
                /*
                String img_saved_path_output="output.png";
                ImageUtils.saveBitmap(cropCopyBitmap, img_saved_path_output);
                */
                // 利用反向矩阵，结果反算为原始图像中的坐标
                cropToFrameTransform.mapRect(location);
                // 重新设置结果
                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            // 跟踪结果
            tracker.trackResults(mappedRecognitions, currTimestamp);
            // 更新浮层
            trackingOverlay.postInvalidate();
            // 关闭检测锁
            computingDetection = false;

            // 实时更新结果以及模型预测耗时
            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // 调用 tensorflow 检测模型
  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    // () -> 为java的匿名函数语法，这里返回的是Runnable类型
    // 这里表示是否使用NNAPI
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    // 表示设置线程数目
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
