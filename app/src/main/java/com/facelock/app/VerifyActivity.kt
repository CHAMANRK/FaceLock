package com.facelock.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.gms.tasks.Tasks

/**
 * Confirmation: LIVE camera only, there is deliberately no way to pick a photo here.
 * With "blink check" on, the face must be seen with open eyes, then closed, then open again
 * before the match is accepted — a basic guard against a held-up photo (not a full anti-spoof).
 */
class VerifyActivity : AppCompatActivity() {
    private companion object { const val WINDOW = 5 }

    private lateinit var engine: FaceEngine
    private lateinit var store: FaceStore
    private lateinit var prefs: Prefs
    private lateinit var previewView: PreviewView
    private lateinit var result: TextView
    private lateinit var hint: TextView
    private var camera: CameraHelper? = null
    private val detector = FaceDetect.createDetector()

    private enum class Blink { WAIT_OPEN, WAIT_CLOSED, WAIT_REOPEN, DONE }
    private var blink = Blink.WAIT_OPEN
    private var done = false
    private val recent = ArrayDeque<Float>()

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else hint.setText(R.string.camera_denied)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_verify)
        engine = FaceEngine.get(this)
        store = FaceStore(this)
        prefs = Prefs(this)
        previewView = findViewById(R.id.preview)
        result = findViewById(R.id.result)
        hint = findViewById(R.id.hint)
        blink = if (prefs.requireBlink) Blink.WAIT_OPEN else Blink.DONE
        hint.setText(if (prefs.requireBlink) R.string.look_and_blink else R.string.look_at_camera)

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        camera = CameraHelper(this, previewView) { frame -> analyse(frame) }.also { it.start() }
    }

    // Runs on the camera executor thread, one frame at a time.
    private fun analyse(frame: android.graphics.Bitmap) {
        if (done) return
        val faces = try {
            Tasks.await(detector.process(FaceDetect.inputImage(frame)))
        } catch (e: Exception) {
            return
        }
        val face = FaceDetect.largest(faces)
        if (face == null) {
            recent.clear()
            ui { result.setTextColor(Color.GRAY); result.setText(R.string.no_face_in_view) }
            return
        }
        if (face.boundingBox.width() < 90 || !FaceDetect.isFrontal(face, 25f, 25f)) {
            recent.clear()
            ui { result.setTextColor(Color.GRAY); result.setText(R.string.look_straight) }
            return
        }

        if (blink != Blink.DONE) {
            val l = face.leftEyeOpenProbability
            val r = face.rightEyeOpenProbability
            if (l != null && r != null) {
                val open = l > 0.6f && r > 0.6f
                val closed = l < 0.3f && r < 0.3f
                when (blink) {
                    Blink.WAIT_OPEN -> if (open) blink = Blink.WAIT_CLOSED
                    Blink.WAIT_CLOSED -> if (closed) blink = Blink.WAIT_REOPEN
                    Blink.WAIT_REOPEN -> if (open) blink = Blink.DONE
                    Blink.DONE -> {}
                }
            }
            if (blink != Blink.DONE) {
                ui { result.setTextColor(Color.GRAY); result.setText(R.string.please_blink) }
                return
            }
        }

        val aligned = FaceDetect.align(frame, face)
        if (aligned == null) {
            recent.clear()
            ui { result.setTextColor(Color.GRAY); result.setText(R.string.no_landmarks) }
            return
        }
        val spec = engine.activeModel(prefs.modelChoice)
        val probe = try {
            engine.embedder(spec.id).embed(aligned)
        } catch (e: Exception) {
            return
        }
        val score = Matcher.score(store.load(), spec.id, probe)
        if (score == null) {
            ui { result.setTextColor(Color.RED); result.setText(R.string.enroll_first) }
            return
        }

        // Decide on several consecutive frames, not one lucky frame.
        recent.addLast(score)
        if (recent.size > WINDOW) recent.removeFirst()
        val threshold = spec.threshold + prefs.strictness
        if (recent.size < WINDOW) {
            ui {
                result.setTextColor(Color.GRAY)
                result.text = getString(R.string.checking_fmt, recent.size, WINDOW)
            }
            return
        }
        val avg = recent.average().toFloat()
        val worst = recent.minOrNull() ?: 0f
        if (avg >= threshold && worst >= threshold - 0.10f) {
            done = true
            ui {
                result.setTextColor(Color.rgb(0, 150, 60))
                result.text = getString(R.string.unlocked_fmt, avg, threshold, spec.label)
                hint.setText(R.string.unlocked)
            }
        } else {
            ui {
                result.setTextColor(Color.RED)
                result.text = getString(R.string.no_match_fmt, avg, threshold, spec.label)
            }
        }
    }

    private fun ui(block: () -> Unit) = runOnUiThread(block)

    override fun onDestroy() {
        super.onDestroy()
        camera?.stop()
        detector.close()
    }
}
