package com.facelock.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlin.math.max
import kotlin.math.min

object FaceDetect {
    /** Accurate mode + eye-open probabilities (needed for the blink check). */
    fun createDetector(): FaceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.25f)
            .build()
    )

    fun largest(faces: List<Face>): Face? = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }

    /** Square crop around the face with some margin, clamped to the image. */
    fun crop(src: Bitmap, box: Rect, margin: Float = 0.25f): Bitmap {
        val cx = box.centerX()
        val cy = box.centerY()
        val half = (max(box.width(), box.height()) * (1f + margin)) / 2f
        val left = max(0, (cx - half).toInt())
        val top = max(0, (cy - half).toInt())
        val right = min(src.width, (cx + half).toInt())
        val bottom = min(src.height, (cy + half).toInt())
        return Bitmap.createBitmap(src, left, top, max(1, right - left), max(1, bottom - top))
    }

    fun rotate(src: Bitmap, degrees: Int): Bitmap {
        if (degrees == 0) return src
        val m = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, m, true)
    }

    /** Decodes a gallery image (downscaled, EXIF-rotated) so it is safe to process. */
    fun loadFromUri(ctx: Context, uri: Uri, maxSide: Int = 1280): Bitmap? {
        val resolver = ctx.contentResolver
        val bounds = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
        resolver.openInputStream(uri)?.use { android.graphics.BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0) return null
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / sample > maxSide * 2) sample *= 2
        val opts = android.graphics.BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val bmp = resolver.openInputStream(uri)?.use { android.graphics.BitmapFactory.decodeStream(it, null, opts) }
            ?: return null
        val exif = resolver.openInputStream(uri)?.use { ExifInterface(it) }
        val degrees = when (exif?.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90
            ExifInterface.ORIENTATION_ROTATE_180 -> 180
            ExifInterface.ORIENTATION_ROTATE_270 -> 270
            else -> 0
        }
        val rotated = rotate(bmp, degrees)
        val scale = maxSide.toFloat() / max(rotated.width, rotated.height)
        return if (scale < 1f) {
            Bitmap.createScaledBitmap(rotated, (rotated.width * scale).toInt(), (rotated.height * scale).toInt(), true)
        } else rotated
    }

    fun inputImage(bmp: Bitmap): InputImage = InputImage.fromBitmap(bmp, 0)
}
