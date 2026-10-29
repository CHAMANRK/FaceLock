package com.facelock.app

import android.content.Context

class Prefs(ctx: Context) {
    private val sp = ctx.getSharedPreferences("facelock", Context.MODE_PRIVATE)

    /** "auto" or a model id from models.json. */
    var modelChoice: String
        get() = sp.getString("model", "auto") ?: "auto"
        set(v) = sp.edit().putString("model", v).apply()

    /** Added to the model's threshold. Positive = stricter. */
    var strictness: Float
        get() = sp.getFloat("strictness", 0f)
        set(v) = sp.edit().putFloat("strictness", v).apply()

    var requireBlink: Boolean
        get() = sp.getBoolean("blink", true)
        set(v) = sp.edit().putBoolean("blink", v).apply()
}
