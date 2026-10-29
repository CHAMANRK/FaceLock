package com.facelock.app

import android.content.Intent
import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.SeekBar
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    private lateinit var prefs: Prefs
    private lateinit var store: FaceStore
    private lateinit var status: TextView
    private lateinit var strictLabel: TextView
    private lateinit var spinner: Spinner
    private lateinit var choices: List<String> // "auto" + model ids

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        prefs = Prefs(this)
        store = FaceStore(this)
        status = findViewById(R.id.status)
        strictLabel = findViewById(R.id.strictLabel)
        spinner = findViewById(R.id.modelSpinner)

        val engine = FaceEngine.get(this)
        if (engine.models.isEmpty()) {
            status.text = getString(R.string.no_models)
            findViewById<Button>(R.id.btnEnroll).isEnabled = false
            findViewById<Button>(R.id.btnUnlock).isEnabled = false
            return
        }

        val auto = engine.activeModel("auto")
        choices = listOf("auto") + engine.models.map { it.id }
        val labels = listOf("Auto — best from benchmark (${auto.label})") + engine.models.map { it.label }
        spinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, labels)
        spinner.setSelection(choices.indexOf(prefs.modelChoice).coerceAtLeast(0))
        spinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p: AdapterView<*>?, v: View?, pos: Int, id: Long) {
                prefs.modelChoice = choices[pos]
            }
            override fun onNothingSelected(p: AdapterView<*>?) {}
        }

        val seek = findViewById<SeekBar>(R.id.strictSeek)
        // progress 0..40 -> strictness -0.20 .. +0.20
        seek.progress = ((prefs.strictness + 0.2f) * 100).toInt().coerceIn(0, 40)
        updateStrictLabel()
        seek.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar?, p: Int, fromUser: Boolean) {
                prefs.strictness = p / 100f - 0.2f
                updateStrictLabel()
            }
            override fun onStartTrackingTouch(s: SeekBar?) {}
            override fun onStopTrackingTouch(s: SeekBar?) {}
        })

        val blink = findViewById<Switch>(R.id.blinkSwitch)
        blink.isChecked = prefs.requireBlink
        blink.setOnCheckedChangeListener { _, on -> prefs.requireBlink = on }

        findViewById<Button>(R.id.btnEnroll).setOnClickListener {
            startActivity(Intent(this, EnrollActivity::class.java))
        }
        findViewById<Button>(R.id.btnUnlock).setOnClickListener {
            if (store.count() == 0) {
                AlertDialog.Builder(this).setMessage(R.string.enroll_first).setPositiveButton(android.R.string.ok, null).show()
            } else startActivity(Intent(this, VerifyActivity::class.java))
        }
        findViewById<Button>(R.id.btnClear).setOnClickListener {
            AlertDialog.Builder(this)
                .setMessage(R.string.confirm_clear)
                .setPositiveButton(android.R.string.ok) { _, _ -> store.clear(); refresh() }
                .setNegativeButton(android.R.string.cancel, null)
                .show()
        }
    }

    private fun updateStrictLabel() {
        strictLabel.text = getString(R.string.strictness_fmt, prefs.strictness)
    }

    override fun onResume() {
        super.onResume()
        if (FaceEngine.get(this).models.isNotEmpty()) refresh()
    }

    private fun refresh() {
        val samples = store.load()
        val live = samples.count { it.source == "live" }
        status.text = getString(R.string.status_fmt, samples.size, live, samples.size - live)
    }
}
