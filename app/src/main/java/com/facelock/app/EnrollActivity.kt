package com.facelock.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.google.android.gms.tasks.Tasks
import java.util.concurrent.Executors

/**
 * Training / enrolment. Two ways to add a face:
 *  1. live: take a photo with the front camera
 *  2. gallery: pick an existing photo
 * Only embeddings are stored (every available model is run, so the model can be switched later).
 */
class EnrollActivity : AppCompatActivity() {
    private lateinit var store: FaceStore
    private lateinit var engine: FaceEngine
    private lateinit var info: TextView
    private lateinit var previewView: PreviewView
    private var camera: CameraHelper? = null
    private val detector = FaceDetect.createDetector()
    private val worker = Executors.newSingleThreadExecutor()

    @Volatile private var lastFrame: Bitmap? = null

    private val askCamera = registerForActivityResult(ActivityResultContracts.RequestPermission()) { ok ->
        if (ok) startCamera() else info.setText(R.string.camera_denied)
    }
    private val pickImages = registerForActivityResult(ActivityResultContracts.GetMultipleContents()) { uris ->
        if (uris.isNotEmpty()) addFromGallery(uris)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_enroll)
        store = FaceStore(this)
        engine = FaceEngine.get(this)
        info = findViewById(R.id.enrollInfo)
        previewView = findViewById(R.id.preview)

        findViewById<Button>(R.id.btnCapture).setOnClickListener { captureLive() }
        findViewById<Button>(R.id.btnGallery).setOnClickListener { pickImages.launch("image/*") }
        updateInfo()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else askCamera.launch(Manifest.permission.CAMERA)
    }

    private fun startCamera() {
        camera = CameraHelper(this, previewView) { lastFrame = it }.also { it.start() }
    }

    private fun updateInfo() {
        val s = store.load()
        val live = s.count { it.source == "live" }
        info.text = getString(R.string.enroll_info_fmt, s.size, live, s.size - live)
    }

    private fun captureLive() {
        val frame = lastFrame ?: return toast(R.string.no_frame)
        process(frame, "live")
    }

    private fun addFromGallery(uris: List<Uri>) {
        worker.execute {
            var added = 0
            for (uri in uris) {
                val bmp = FaceDetect.loadFromUri(this, uri) ?: continue
                if (enrol(bmp, "gallery")) added++
            }
            runOnUiThread {
                toast(getString(R.string.gallery_result_fmt, added, uris.size))
                updateInfo()
            }
        }
    }

    private fun process(bmp: Bitmap, source: String) {
        worker.execute {
            val ok = enrol(bmp, source)
            runOnUiThread {
                toast(if (ok) getString(R.string.face_added) else getString(R.string.no_face_found))
                updateInfo()
            }
        }
    }

    /** Detects the biggest face, embeds it with every model and stores it. Must run off the main thread. */
    private fun enrol(bmp: Bitmap, source: String): Boolean {
        return try {
            val faces = Tasks.await(detector.process(FaceDetect.inputImage(bmp)))
            val face = FaceDetect.largest(faces) ?: return false
            val crop = FaceDetect.crop(bmp, face.boundingBox)
            store.add(FaceStore.Sample(source, System.currentTimeMillis(), engine.embedAll(crop)))
            true
        } catch (e: Exception) {
            false
        }
    }

    private fun toast(res: Int) = Toast.makeText(this, res, Toast.LENGTH_SHORT).show()
    private fun toast(s: String) = Toast.makeText(this, s, Toast.LENGTH_SHORT).show()

    override fun onDestroy() {
        super.onDestroy()
        camera?.stop()
        detector.close()
        worker.shutdown()
    }
}
