package com.facelock.app

import android.content.Context
import org.json.JSONObject

/** One face-embedding model, described in assets/models.json. */
data class ModelSpec(
    val id: String,
    val label: String,
    val file: String,
    /** "fixed" = (x - mean) / std, "per_image" = standardise each image by its own mean/std. */
    val norm: String,
    val mean: Float,
    val std: Float,
    /** Cosine-similarity needed to count as the same person. */
    val threshold: Float,
)

data class ModelConfig(val defaultId: String, val models: List<ModelSpec>)

object ModelRegistry {
    /** Reads models.json and keeps only the models whose .tflite file is actually in the APK. */
    fun load(ctx: Context): ModelConfig {
        val text = ctx.assets.open("models.json").bufferedReader().use { it.readText() }
        val json = JSONObject(text)
        val arr = json.getJSONArray("models")
        val present = ctx.assets.list("models")?.toSet() ?: emptySet()
        val list = ArrayList<ModelSpec>()
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val file = o.getString("file")
            if (file !in present) continue
            list.add(
                ModelSpec(
                    id = o.getString("id"),
                    label = o.optString("label", o.getString("id")),
                    file = file,
                    norm = o.optString("norm", "fixed"),
                    mean = o.optDouble("mean", 127.5).toFloat(),
                    std = o.optDouble("std", 128.0).toFloat(),
                    threshold = o.optDouble("threshold", 0.5).toFloat(),
                )
            )
        }
        return ModelConfig(json.optString("default", ""), list)
    }
}
