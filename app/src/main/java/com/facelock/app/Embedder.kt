package com.facelock.app

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Runs one float32 NHWC face-embedding .tflite model.
 * Input size and batch size are read from the model itself (MobileFaceNet wants batch = 2),
 * so the same face is simply repeated across the batch and only the first output is used.
 */
class Embedder(ctx: Context, val spec: ModelSpec) : Closeable {
    private val interpreter: Interpreter
    private val batch: Int
    private val size: Int
    private val outDim: Int

    init {
        val mapped = ctx.assets.openFd("models/${spec.file}").use { fd ->
            FileInputStream(fd.fileDescriptor).channel
                .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        }
        interpreter = Interpreter(mapped, Interpreter.Options().setNumThreads(4))
        val inShape = interpreter.getInputTensor(0).shape() // [batch, h, w, 3]
        batch = inShape[0]
        size = inShape[1]
        outDim = interpreter.getOutputTensor(0).shape().last()
    }

    /** Returns an L2-normalised embedding for an (already cropped) face. */
    @Synchronized
    fun embed(face: Bitmap): FloatArray {
        val scaled = Bitmap.createScaledBitmap(face, size, size, true)
        val px = IntArray(size * size)
        scaled.getPixels(px, 0, size, 0, 0, size, size)

        val one = FloatArray(size * size * 3)
        for (i in px.indices) {
            val p = px[i]
            one[i * 3] = ((p shr 16) and 0xFF).toFloat()
            one[i * 3 + 1] = ((p shr 8) and 0xFF).toFloat()
            one[i * 3 + 2] = (p and 0xFF).toFloat()
        }
        normalise(one)

        val buf = ByteBuffer.allocateDirect(4 * batch * one.size).order(ByteOrder.nativeOrder())
        val fb = buf.asFloatBuffer()
        repeat(batch) { fb.put(one) }

        val out = Array(batch) { FloatArray(outDim) }
        interpreter.run(buf, out)
        return l2(out[0])
    }

    private fun normalise(a: FloatArray) {
        if (spec.norm == "per_image") {
            var sum = 0.0
            for (v in a) sum += v
            val mean = (sum / a.size).toFloat()
            var sq = 0.0
            for (v in a) {
                val d = v - mean
                sq += d * d
            }
            val std = max(sqrt(sq / a.size).toFloat(), 1f / sqrt(a.size.toFloat()))
            for (i in a.indices) a[i] = (a[i] - mean) / std
        } else {
            for (i in a.indices) a[i] = (a[i] - spec.mean) / spec.std
        }
    }

    private fun l2(v: FloatArray): FloatArray {
        var s = 0.0
        for (x in v) s += x * x
        val n = (sqrt(s) + 1e-10).toFloat()
        return FloatArray(v.size) { v[it] / n }
    }

    override fun close() = interpreter.close()
}
