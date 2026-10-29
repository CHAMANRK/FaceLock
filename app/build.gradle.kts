plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.facelock.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.facelock.app"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.2.0"

        // phones are arm64; keeps the APK small (ONNX Runtime ships one .so per ABI)
        ndk { abiFilters += listOf("arm64-v8a") }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.exifinterface:exifinterface:1.3.7")

    // Camera
    val camerax = "1.3.4"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // Face detection (face box, landmarks for alignment, head angles, eye-open probabilities for the blink check)
    implementation("com.google.mlkit:face-detection:16.1.6")
    implementation("com.google.android.gms:play-services-tasks:18.2.0")

    // Face embedding model (ArcFace, .onnx)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
}
