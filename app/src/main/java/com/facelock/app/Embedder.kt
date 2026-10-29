package com.facelock.app

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import java.io.Closeable
import java.nio.FloatBuffer
import kotlin.math.sqrt

/**
 * Runs one ArcFace-style ONNX model: input [1,3,S,S] RGB (NCHW), output [1,D].
 * The bitmap passed to [embed] must already be an ALIGNED face (see FaceDetect.align).
 */
class Embedder(ctx: Context, val spec: ModelSpec) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val size: Int

    init {
        val bytes = ctx.assets.open("models/${spec.file}").use { it.readBytes() }
        val opts = OrtSession.SessionOptions().apply { setIntraOpNumThreads(4) }
        session = env.createSession(bytes, opts)
        inputName = session.inputNames.first()
        val shape = (session.inputInfo[inputName]!!.info as TensorInfo).shape
        size = shape.last().toInt().takeIf { it > 0 } ?: 112
    }

    /** Returns an L2-normalised embedding for an aligned face. */
    @Synchronized
    @Suppress("UNCHECKED_CAST")
    fun embed(aligned: Bitmap): FloatArray {
        val bmp = if (aligned.width == size && aligned.height == size) aligned
        else Bitmap.createScaledBitmap(aligned, size, size, true)
        val px = IntArray(size * size)
        bmp.getPixels(px, 0, size, 0, 0, size, size)

        val plane = size * size
        val data = FloatArray(3 * plane)
        for (i in px.indices) {
            val p = px[i]
            data[i] = (((p shr 16) and 0xFF) - spec.mean) / spec.std            // R
            data[plane + i] = (((p shr 8) and 0xFF) - spec.mean) / spec.std     // G
            data[2 * plane + i] = ((p and 0xFF) - spec.mean) / spec.std         // B
        }

        val shape = longArrayOf(1, 3, size.toLong(), size.toLong())
        return OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape).use { tensor ->
            session.run(mapOf(inputName to tensor)).use { res ->
                l2((res[0].value as Array<FloatArray>)[0])
            }
        }
    }

    private fun l2(v: FloatArray): FloatArray {
        var s = 0.0
        for (x in v) s += x * x
        val n = (sqrt(s) + 1e-10).toFloat()
        return FloatArray(v.size) { v[it] / n }
    }

    override fun close() {
        session.close()
    }
}
