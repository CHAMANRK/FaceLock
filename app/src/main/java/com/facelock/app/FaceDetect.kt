package com.facelock.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import com.google.mlkit.vision.face.FaceLandmark
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

object FaceDetect {
    /** Accurate mode + landmarks (for alignment) + eye-open probabilities (for the blink check). */
    fun createDetector(): FaceDetector = FaceDetection.getClient(
        FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
            .setMinFaceSize(0.2f)
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


    /** Face must look roughly straight at the camera (ArcFace is trained on near-frontal faces). */
    fun isFrontal(face: Face, maxYaw: Float = 20f, maxPitch: Float = 20f): Boolean =
        abs(face.headEulerAngleY) <= maxYaw && abs(face.headEulerAngleX) <= maxPitch

    // Standard ArcFace 112x112 template: eyes + mouth corners.
    private const val OUT = 112
    private val TEMPLATE = floatArrayOf(
        38.2946f, 51.6963f,   // left eye (image-left)
        73.5318f, 51.5014f,   // right eye
        41.5493f, 92.3655f,   // mouth, image-left corner
        70.7299f, 92.2041f,   // mouth, image-right corner
    )

    /**
     * Warps the face to the canonical 112x112 ArcFace pose using a similarity transform
     * (rotation + scale + shift) fitted on eyes and mouth corners. Returns null if landmarks are missing.
     * Points are ordered by x so it does not matter how ML Kit labels left/right.
     */
    fun align(src: Bitmap, face: Face): Bitmap? {
        fun pt(type: Int) = face.getLandmark(type)?.position
        val le = pt(FaceLandmark.LEFT_EYE) ?: return null
        val re = pt(FaceLandmark.RIGHT_EYE) ?: return null
        val ml = pt(FaceLandmark.MOUTH_LEFT) ?: return null
        val mr = pt(FaceLandmark.MOUTH_RIGHT) ?: return null
        val eyes = listOf(le, re).sortedBy { it.x }
        val mouth = listOf(ml, mr).sortedBy { it.x }
        val s = floatArrayOf(
            eyes[0].x, eyes[0].y, eyes[1].x, eyes[1].y,
            mouth[0].x, mouth[0].y, mouth[1].x, mouth[1].y,
        )
        val n = 4
        var mx = 0.0; var my = 0.0; var mu = 0.0; var mv = 0.0
        for (i in 0 until n) { mx += s[2 * i]; my += s[2 * i + 1]; mu += TEMPLATE[2 * i]; mv += TEMPLATE[2 * i + 1] }
        mx /= n; my /= n; mu /= n; mv /= n
        var numA = 0.0; var numB = 0.0; var den = 0.0
        for (i in 0 until n) {
            val xs = s[2 * i] - mx; val ys = s[2 * i + 1] - my
            val xd = TEMPLATE[2 * i] - mu; val yd = TEMPLATE[2 * i + 1] - mv
            numA += xs * xd + ys * yd
            numB += xs * yd - ys * xd
            den += xs * xs + ys * ys
        }
        if (den < 1e-6) return null
        val a = numA / den
        val b = numB / den
        val tx = mu - (a * mx - b * my)
        val ty = mv - (b * mx + a * my)
        val m = Matrix().apply {
            setValues(floatArrayOf(a.toFloat(), (-b).toFloat(), tx.toFloat(), b.toFloat(), a.toFloat(), ty.toFloat(), 0f, 0f, 1f))
        }
        val out = Bitmap.createBitmap(OUT, OUT, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(src, m, Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG))
        return out
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
