// Verify TensorFlow.js is loaded
console.log('TensorFlow.js version:', tf.version.tfjs);

// Global variables for feedback timing
let lastFeedbackTime = 0;
const feedbackInterval = 30000; // 30 seconds in milliseconds
let detector = null; // Hold the detector globally
let animationFrameId = null; // To control the animation loop

// Get elements needed multiple times
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const feedback = document.getElementById('feedback');
const startButton = document.getElementById('start-button');

// Load the Pose Detection model and set up the video feed
async function setupCamera() {
    console.log("Requesting camera access...");
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;

        // Provide immediate feedback that camera is trying to load
        feedback.innerHTML = "Activating camera...";
        console.log("Camera stream acquired");

        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                console.log("Video metadata loaded");
                // Set canvas size only once metadata is loaded
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                 console.log(`Canvas dimensions set to: ${canvas.width}x${canvas.height}`);

                // IMMEDIATE FEEDBACK: Camera is ready
                feedback.innerHTML = "Camera activated. Preparing analysis...";
                resolve(video);
            };
             video.onerror = () => {
                console.error("Error loading video metadata.");
                feedback.innerHTML = "Error loading video.";
             }
        });
    } catch (error) {
        console.error('Error accessing the camera:', error);
        feedback.innerHTML = 'Error: Unable to access camera. Please check permissions and reload.';
        alert('Unable to access the camera. Please check your browser permissions and try again.');
        return null; // Return null if camera setup fails
    }
}

// Load the Pose Detection model
async function loadModel() {
     if (!detector) { // Only load if not already loaded
        console.log("Loading MoveNet model...");
        feedback.innerHTML = "Loading posture analysis model...";
        try {
             // Use MoveNet Lightning for speed, or Thunder for accuracy
            const detectorConfig = {modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING};
            detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
            console.log("MoveNet model loaded successfully!");
            feedback.innerHTML = "Model loaded. Ready to detect posture.";
        } catch(error) {
            console.error("Error loading the model:", error);
            feedback.innerHTML = "Error loading analysis model.";
            detector = null; // Ensure detector is null if loading fails
        }
    }
     return detector;
}

// Detect poses in real-time
async function detectPose() {
    if (!detector || video.readyState < 2) { // Ensure detector is loaded and video is ready
        console.log("Detector or video not ready, skipping detection frame.");
         // Request next frame slightly delayed to allow setup
         animationFrameId = requestAnimationFrame(detectPose);
        return;
    }

    try {
        const poses = await detector.estimatePoses(video, {
             flipHorizontal: false // Input already flipped via CSS transform: scaleX(-1)
        });
        const now = Date.now(); // Get current time

        // Clear the canvas before drawing
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (poses.length > 0) {
            const keypoints = poses[0].keypoints;

             // Ensure keypoints are scaled correctly if canvas/video differ - usually not needed if sizes match
             // const scaleX = canvas.width / video.videoWidth;
             // const scaleY = canvas.height / video.videoHeight;
             // const scaledKeypoints = keypoints.map(kp => ({...kp, x: kp.x * scaleX, y: kp.y * scaleY}));

            // Draw keypoints and skeleton (every frame for visual feedback)
             // Pass original keypoints if not scaling, or scaledKeypoints if you are
            drawKeypoints(ctx, keypoints);
            drawSkeleton(ctx, keypoints);

            // --- TIMED FEEDBACK: Check time before giving posture feedback ---
            if (now - lastFeedbackTime >= feedbackInterval) {
                console.log("Generating feedback - interval passed.");
                const postureFeedback = generateFeedback(keypoints);
                feedback.innerHTML = postureFeedback; // Update posture feedback div
                lastFeedbackTime = now; // Reset the timer
            }
             // --- End of timed feedback check ---

        } else {
            // Optional: If no poses are detected after a while, update feedback?
             // Only update if the interval has passed to avoid flickering messages
             if (now - lastFeedbackTime >= feedbackInterval) {
                 if (!feedback.innerHTML.includes("No person detected")) { // Avoid repetition
                    feedback.innerHTML = "No person detected in view.";
                    console.log("No poses detected.");
                 }
                 lastFeedbackTime = now; // Still reset timer
             }
        }
    } catch (error) {
        console.error("Error during pose estimation:", error);
        // Potentially stop the loop or show an error message
        // feedback.innerHTML = "An error occurred during analysis.";
        // cancelAnimationFrame(animationFrameId); // Example: stop on error
    }


    // Loop the detection
    animationFrameId = requestAnimationFrame(detectPose);
}

// Draw keypoints on the canvas
function drawKeypoints(ctx, keypoints) {
    const keypointRadius = 5; // Smaller radius
    ctx.fillStyle = "red";
    keypoints.forEach((keypoint) => {
        if (keypoint.score > 0.5) { // Confidence threshold
            ctx.beginPath();
            ctx.arc(keypoint.x, keypoint.y, keypointRadius, 0, 2 * Math.PI);
            ctx.fill();
        }
    });
}

// Draw skeleton on the canvas
function drawSkeleton(ctx, keypoints) {
    ctx.strokeStyle = "blue";
    ctx.lineWidth = 2; // Thinner lines
    const adjacentPairs = poseDetection.util.getAdjacentPairs(poseDetection.SupportedModels.MoveNet);

    adjacentPairs.forEach(([i, j]) => {
        const kp1 = keypoints[i];
        const kp2 = keypoints[j];
        if (kp1.score > 0.5 && kp2.score > 0.5) { // Confidence threshold
            ctx.beginPath();
            ctx.moveTo(kp1.x, kp1.y);
            ctx.lineTo(kp2.x, kp2.y);
            ctx.stroke();
        }
    });
}

// Generate feedback based on keypoints
function generateFeedback(keypoints) {
    let feedbackMessages = []; // Store messages in an array

    // --- INCREASED TOLERANCE THRESHOLDS ---
    const shoulderAlignmentThreshold = 35; // Increased tolerance
    const neckCenteringThreshold = 35;   // Increased tolerance
    const headTiltThreshold = 30;       // Increased tolerance

    // Find keypoints needed for upper body checks
    const leftShoulder = keypoints.find((kp) => kp.name === 'left_shoulder');
    const rightShoulder = keypoints.find((kp) => kp.name === 'right_shoulder');
    const leftEar = keypoints.find((kp) => kp.name === 'left_ear');
    const rightEar = keypoints.find((kp) => kp.name === 'right_ear');
    const nose = keypoints.find((kp) => kp.name === 'nose');

    // Check shoulder alignment
    if (leftShoulder && rightShoulder && leftShoulder.score > 0.5 && rightShoulder.score > 0.5) {
        const shoulderDiff = Math.abs(leftShoulder.y - rightShoulder.y);
        if (shoulderDiff > shoulderAlignmentThreshold) { // Use new threshold
            feedbackMessages.push("Align shoulders evenly.");
        }
    } else {
         console.log("Skipping shoulder check - keypoints missing or low confidence.");
    }

    // Check neck alignment (head centered over shoulders)
    if (nose && leftShoulder && rightShoulder && nose.score > 0.5 && leftShoulder.score > 0.5 && rightShoulder.score > 0.5) {
        const neckCenter = (leftShoulder.x + rightShoulder.x) / 2;
        const neckOffset = Math.abs(nose.x - neckCenter);
        if (neckOffset > neckCenteringThreshold) { // Use new threshold
            feedbackMessages.push("Center your head over your shoulders.");
        }
    } else {
         console.log("Skipping neck centering check - keypoints missing or low confidence.");
    }

    // Check head tilt (ears level)
    if (leftEar && rightEar && leftEar.score > 0.5 && rightEar.score > 0.5) {
        const earDiff = Math.abs(leftEar.y - rightEar.y);
        if (earDiff > headTiltThreshold) { // Use new threshold
            feedbackMessages.push("Keep your head level (avoid tilting).");
        }
    } else {
         console.log("Skipping head tilt check - keypoints missing or low confidence.");
    }

    // --- Compile feedback ---
    if (feedbackMessages.length === 0) {
        // Only return "Good posture" if checks were likely performed (keypoints existed)
         if (leftShoulder && rightShoulder && leftEar && rightEar && nose) {
             return "Good posture! Keep it up.";
         } else {
             // If keypoints were missing, maybe avoid saying "Good posture"
             return "Keep still for analysis."; // Or return last known feedback
         }
    } else {
        return feedbackMessages.join(" "); // Join multiple messages with a space
    }
}


// Main function to orchestrate setup and detection
async function startPosturePractice() {
     startButton.disabled = true; // Disable button while setting up
     startButton.innerHTML = "Starting...";

     // 1. Setup Camera
     const videoElement = await setupCamera();
     if (!videoElement) {
         startButton.disabled = false; // Re-enable button if setup fails
         startButton.innerHTML = "Start Practice";
         return; // Stop if camera failed
     }
     videoElement.play(); // Ensure video plays

     // 2. Load Model
     await loadModel(); // Wait for the model to load
      if (!detector) {
          startButton.disabled = false; // Re-enable button if model fails
          startButton.innerHTML = "Start Practice";
          feedback.innerHTML = "Failed to load analysis model. Please try again."
          return; // Stop if model failed
      }


     // 3. Start Detection Loop
     lastFeedbackTime = Date.now() - feedbackInterval + 5000; // Give feedback soon after start
     console.log("Starting detection loop...");
     feedback.innerHTML = "Starting posture analysis...";

     // Clear previous loop if any
     if (animationFrameId) {
         cancelAnimationFrame(animationFrameId);
     }
     detectPose(); // Start the loop

      // Update button state - maybe change to a "Stop" button?
      // For now, just indicate it's running.
      // startButton.innerHTML = "Running...";
      // startButton.disabled = true; // Keep disabled while running
}


// Start the session when the button is clicked
startButton.addEventListener('click', startPosturePractice);

// Optional: Add cleanup logic if needed (e.g., stop camera stream when leaving page)
// window.addEventListener('beforeunload', () => {
//     if (video.srcObject) {
//         video.srcObject.getTracks().forEach(track => track.stop());
//     }
//     if (animationFrameId) {
//         cancelAnimationFrame(animationFrameId);
//     }
// });

