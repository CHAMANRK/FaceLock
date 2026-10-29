package com.facelock.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** Keeps enrolled face embeddings (no photos) in the app's private storage. */
class FaceStore(ctx: Context) {
    class Sample(val source: String, val timestamp: Long, val emb: Map<String, FloatArray>)

    private val file = File(ctx.filesDir, "faces.json")

    @Synchronized
    fun load(): List<Sample> {
        if (!file.exists()) return emptyList()
        return try {
            val arr = JSONObject(file.readText()).getJSONArray("samples")
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                val e = o.getJSONObject("emb")
                val map = HashMap<String, FloatArray>()
                for (key in e.keys()) {
                    val a = e.getJSONArray(key)
                    map[key] = FloatArray(a.length()) { a.getDouble(it).toFloat() }
                }
                Sample(o.optString("source", "live"), o.optLong("ts"), map)
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    @Synchronized
    fun add(sample: Sample) {
        save(load() + sample)
    }

    @Synchronized
    fun count(): Int = load().size

    @Synchronized
    fun clear() {
        file.delete()
    }

    private fun save(samples: List<Sample>) {
        val arr = JSONArray()
        for (s in samples) {
            val emb = JSONObject()
            for ((k, v) in s.emb) {
                val a = JSONArray()
                for (x in v) a.put(x.toDouble())
                emb.put(k, a)
            }
            arr.put(JSONObject().put("source", s.source).put("ts", s.timestamp).put("emb", emb))
        }
        val tmp = File(file.parentFile, "faces.json.tmp")
        tmp.writeText(JSONObject().put("samples", arr).toString())
        if (!tmp.renameTo(file)) {
            file.writeText(tmp.readText())
            tmp.delete()
        }
    }
}
