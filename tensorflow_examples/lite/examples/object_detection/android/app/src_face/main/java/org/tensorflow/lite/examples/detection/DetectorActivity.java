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
// 检测的Activity继承自相机Activity，并且要实现OnImageAvailableListener
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
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

// 设置detection模型输入参数，包括图像长宽
private static final int TF_OD_API_INPUT_SIZE = 300;
private static final int TF_OD_API_INPUT_HEIGHT = 240;
private static final int TF_OD_API_INPUT_WIDTH = 320;

//private static final boolean TF_OD_API_IS_QUANTIZED = false;
//private static final String TF_OD_API_MODEL_FILE = "face_detect_20191128.tflite";
// 是否开启量化
private static final boolean TF_OD_API_IS_QUANTIZED = true;
private static final String TF_OD_API_MODEL_FILE = "face_detect_quantized_20191128.tflite";

//private static final boolean TF_OD_API_IS_QUANTIZED = true;
//private static final String TF_OD_API_MODEL_FILE = "face_detect_quan.tflite";

// 类别映射文件
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/facelabelmap.txt";

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // 检测框的最小confidence
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  // 设置是否保持长宽比
  private static final boolean MAINTAIN_ASPECT = false;
  // 设置预览窗口尺寸
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  //private static final Size DESIRED_PREVIEW_SIZE = new Size(320, 240);
  // 是否保存预览图像bitmap
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  // 设置文字大小
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  // 传感器方向
  private Integer sensorOrientation;

  // 创建检测器
  private Classifier detector;

  // 最近一次处理耗时
  private long lastProcessingTimeMs;
  // 构造输入的bitmap
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  // 获取resize之后的图像
  public Bitmap resizeImage(Bitmap bitmap, int w, int h) {
    Bitmap BitmapOrg = bitmap;
    int width = BitmapOrg.getWidth();
    int height = BitmapOrg.getHeight();
    int newWidth = w;
    int newHeight = h;

    // 获取scale的尺寸
    float scaleWidth = ((float) newWidth) / width;
    float scaleHeight = ((float) newHeight) / height;

    // 按照矩阵来操作实现放缩
    Matrix matrix = new Matrix();
    matrix.postScale(scaleWidth, scaleHeight);
    // if you want to rotate the Bitmap
    // matrix.postRotate(45);
    Bitmap resizedBitmap = Bitmap.createBitmap(BitmapOrg, 0, 0, width,
            height, matrix, true);
    return resizedBitmap;
  }


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

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    // 创建跟踪器
    tracker = new MultiBoxTracker(this);

    //int cropSize = TF_OD_API_INPUT_SIZE;

    int cropHeight = TF_OD_API_INPUT_HEIGHT;
    int cropWidth = TF_OD_API_INPUT_WIDTH;

    try {
      // 根据参数载入模型
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

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    //croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropWidth, cropHeight, Config.ARGB_8888);

    // 获得图像变换矩阵, 图像可能旋转，可能有未保持长宽比的尺寸
    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            //cropSize, cropSize,
                cropWidth, cropHeight,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    // 获得tracking的叠加层
    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
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

    // 设置帧参数
    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    // load bitmap from path
    //String img_path="/storage/emulated/0/tensorflow/bbb.png";
    //rgbFrameBitmap = BitmapFactory.decodeFile(img_path);
      rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    //String img_saved_path="bbb.png";
    //ImageUtils.saveBitmap(rgbFrameBitmap, img_saved_path);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    //canvas.drawBitmap(rgbFrameBitmap,  new Matrix(), null);
    // For examining the actual TF input.
    //croppedBitmap = resizeImage(croppedBitmap, 320, 240);
    //croppedBitmap = getResizedBitmap(croppedBitmap, 320, 240);

    // 是否保存预览的bitmap
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            // 预测结果
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            // 获取预测结果
            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                // 绘制矩形
                canvas.drawRect(location, paint);

                // 矩形框给到transform实例
                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    // 显示预览分辨率与cropCopyBitmap的分辨率
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

// 使用的是哪种检测模型
  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

// 设置并行的线程数
  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }
}
