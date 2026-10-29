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
        versionName = "0.1.0"
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

    // .tflite files must stay uncompressed so they can be memory-mapped from assets
    androidResources {
        noCompress += "tflite"
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

    // Face detection (finds the face + eye-open probabilities for the blink check)
    implementation("com.google.mlkit:face-detection:16.1.6")
    implementation("com.google.android.gms:play-services-tasks:18.2.0")

    // Face embedding models (.tflite)
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
}
