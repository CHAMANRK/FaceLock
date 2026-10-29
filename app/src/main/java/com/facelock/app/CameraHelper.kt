package com.facelock.app

import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/** Front-camera preview + a stream of upright Bitmaps for analysis. */
class CameraHelper(
    private val activity: AppCompatActivity,
    private val previewView: PreviewView,
    private val onFrame: (Bitmap) -> Unit,
) {
    private val executor: ExecutorService = Executors.newSingleThreadExecutor()

    fun start() {
        val future = ProcessCameraProvider.getInstance(activity)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also { it.setSurfaceProvider(previewView.surfaceProvider) }
            val analysis = ImageAnalysis.Builder()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
            analysis.setAnalyzer(executor) { proxy -> handle(proxy) }
            provider.unbindAll()
            provider.bindToLifecycle(activity, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(activity))
    }

    private fun handle(proxy: ImageProxy) {
        try {
            val upright = FaceDetect.rotate(proxy.toBitmap(), proxy.imageInfo.rotationDegrees)
            onFrame(upright)
        } catch (_: Exception) {
        } finally {
            proxy.close()
        }
    }

    fun stop() {
        try {
            ProcessCameraProvider.getInstance(activity).get().unbindAll()
        } catch (_: Exception) {
        }
        executor.shutdown()
    }
}
