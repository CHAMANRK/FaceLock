package com.facelock.app

import android.content.Context
import android.graphics.Bitmap
import android.util.Log

/** Loads every available model once and offers embed / match helpers. */
class FaceEngine private constructor(ctx: Context) {
    val config: ModelConfig = ModelRegistry.load(ctx)

    private val embedders: Map<String, Embedder> =
        config.models.mapNotNull { spec ->
            try {
                spec.id to Embedder(ctx, spec)
            } catch (e: Exception) {
                Log.e(TAG, "Could not load model ${spec.id}", e)
                null
            }
        }.toMap()

    /** Models that were found AND loaded successfully. */
    val models: List<ModelSpec> = config.models.filter { embedders.containsKey(it.id) }

    fun embedder(id: String): Embedder = embedders.getValue(id)

    /** Embeds the face with every model (used when enrolling, so any model can be picked later). */
    fun embedAll(face: Bitmap): Map<String, FloatArray> =
        models.associate { it.id to embedder(it.id).embed(face) }

    /** "auto" = the model chosen by the CI benchmark (models.json "default"). */
    fun activeModel(choice: String): ModelSpec {
        val picked = if (choice != "auto") models.firstOrNull { it.id == choice } else null
        return picked ?: models.firstOrNull { it.id == config.defaultId } ?: models.first()
    }

    companion object {
        private const val TAG = "FaceEngine"

        @Volatile
        private var instance: FaceEngine? = null

        fun get(ctx: Context): FaceEngine =
            instance ?: synchronized(this) {
                instance ?: FaceEngine(ctx.applicationContext).also { instance = it }
            }
    }
}

object Matcher {
    /** Both vectors are L2-normalised, so the dot product is the cosine similarity. */
    fun cosine(a: FloatArray, b: FloatArray): Float {
        var s = 0f
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    /**
     * Mean of the best (up to) 3 similarities against the enrolled samples of this model.
     * Returns null if nothing is enrolled for the model.
     */
    fun score(samples: List<FaceStore.Sample>, modelId: String, probe: FloatArray): Float? {
        val sims = samples.mapNotNull { it.emb[modelId] }
            .filter { it.size == probe.size }
            .map { cosine(it, probe) }
            .sortedDescending()
            .take(3)
        return if (sims.isEmpty()) null else sims.average().toFloat()
    }
}
