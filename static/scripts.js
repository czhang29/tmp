// Global variables
let lastFeedbackTime = 0;
const feedbackInterval = 15000; // 15 seconds in milliseconds
let lastAudioAlertTime = 0;
const audioAlertInterval = 30000; // 30 seconds minimum between audio alerts
let detector = null; // Hold the detector globally
let animationFrameId = null; // To control the animation loop
let countdownTimer = null; // For the initial countdown timer
let practiceTimer = null; // For the practice timer
let countdownSeconds = 5; // Default countdown seconds
let practiceSeconds = 120; // Default practice duration (2 minutes)
let practiceStartTime = null; // When practice actually started
let isAnalyzing = false; // Flag to track if analysis is running
let feedbackHistory = []; // Store feedback history for summary
let practiceFocus = 'all'; // Default to analyze all posture aspects
let audioEnabled = true; // Flag for audio alerts
let feedbackSound = null; // Audio element for feedback sound

// Lower thresholds to trigger more feedback
const THRESHOLDS = {
    // For head and neck position
    neckCenteringThreshold: 20, // Original: 30
    headTiltThreshold: 15,      // Original: 25
    
    // For shoulders position
    shoulderAlignmentThreshold: 20, // Original: 30
    shoulderTensionThreshold: 30,   // Original: 40
    
    // For complete feedback
    completeShoulderAlignmentThreshold: 25, // Original: 35
    completeNeckCenteringThreshold: 25,     // Original: 35
    completeHeadTiltThreshold: 20           // Original: 30
};

// Get elements needed multiple times
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas?.getContext('2d');
const feedback = document.getElementById('feedback');
const startButton = document.getElementById('start-button');
const countdownElement = document.getElementById('countdown');
const endPracticeButton = document.getElementById('end-practice-button');
const practiceTimerElement = document.getElementById('practice-timer');

// Initialize based on page focus and duration (if set)
document.addEventListener('DOMContentLoaded', function() {
    // Setup audio element for feedback
    feedbackSound = new Audio('/static/sounds/ding.mp3');
    
    // Check if we're on the timer selection page
    if (document.getElementById('timer-selection')) {
        setupTimerSelection();
        return; // Exit early if on timer selection page
    }
    
    // Check if we're on the countdown page
    if (document.getElementById('fullscreen-countdown')) {
        startFullscreenCountdown();
        return; // Exit early if on countdown page
    }
    
    // Regular practice page initialization
    
    // Check if practice focus is set on the page
    const focusElement = document.getElementById('practice-focus');
    if (focusElement) {
        practiceFocus = focusElement.getAttribute('data-focus') || 'all';
        console.log(`Practice focus set to: ${practiceFocus}`);
    }
    
    // Check if practice duration is set
    const durationElement = document.getElementById('practice-duration');
    if (durationElement) {
        const durationSeconds = parseInt(durationElement.getAttribute('data-seconds'));
        if (!isNaN(durationSeconds) && durationSeconds > 0) {
            practiceSeconds = durationSeconds;
            console.log(`Practice duration set to: ${practiceSeconds} seconds`);
            
            // Update the timer display
            if (practiceTimerElement) {
                practiceTimerElement.textContent = formatTime(practiceSeconds);
            }
        }
    }
    
    // Setup event listeners
    if (startButton) {
        startButton.addEventListener('click', startPosturePractice);
    }
    
    if (endPracticeButton) {
        endPracticeButton.addEventListener('click', endPracticeSession);
    }
    
    // Setup audio toggle if exists
    const audioToggle = document.getElementById('audio-toggle');
    if (audioToggle) {
        audioToggle.addEventListener('click', toggleAudio);
    }
});

// Setup timer selection
function setupTimerSelection() {
    const timerOptions = document.querySelectorAll('.timer-option');
    
    timerOptions.forEach(option => {
        option.addEventListener('click', function() {
            // Remove selected class from all options
            timerOptions.forEach(opt => opt.classList.remove('selected'));
            
            // Add selected class to clicked option
            this.classList.add('selected');
            
            // Update the hidden input value
            const selectedTime = this.getAttribute('data-seconds');
            document.getElementById('selected-timer').value = selectedTime;
            
            // Enable the continue button
            document.getElementById('continue-button').disabled = false;
        });
    });
}

// Start fullscreen countdown
function startFullscreenCountdown() {
    const countdownElement = document.getElementById('countdown-number');
    const redirectUrl = document.getElementById('redirect-url').value;
    let seconds = 5;
    
    countdownElement.textContent = seconds;
    
    const countdownInterval = setInterval(() => {
        seconds--;
        
        if (seconds <= 0) {
            clearInterval(countdownInterval);
            window.location.href = redirectUrl;
        } else {
            countdownElement.textContent = seconds;
        }
    }, 1000);
}

// Toggle audio alerts
function toggleAudio() {
    audioEnabled = !audioEnabled;
    const audioToggle = document.getElementById('audio-toggle');
    
    if (audioToggle) {
        if (audioEnabled) {
            audioToggle.innerHTML = '<i class="fas fa-volume-up"></i>';
            audioToggle.classList.remove('muted');
        } else {
            audioToggle.innerHTML = '<i class="fas fa-volume-mute"></i>';
            audioToggle.classList.add('muted');
        }
    }
}

// Play audio alert for posture problems
function playPostureAlert() {
    const now = Date.now();
    
    // Only play if enough time has passed since the last alert
    if (audioEnabled && now - lastAudioAlertTime >= audioAlertInterval) {
        feedbackSound.play();
        lastAudioAlertTime = now;
        console.log("Playing posture alert sound");
    }
}

// Helper function to format time as MM:SS
function formatTime(totalSeconds) {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

// Initial countdown timer before practice starts
function startCountdown(seconds, onComplete) {
    if (!countdownElement) return;
    
    countdownElement.style.display = 'block';
    countdownSeconds = seconds;
    
    countdownElement.innerHTML = `Get ready: ${countdownSeconds}`;
    
    countdownTimer = setInterval(() => {
        countdownSeconds--;
        
        if (countdownSeconds <= 0) {
            clearInterval(countdownTimer);
            countdownElement.style.display = 'none';
            
            if (typeof onComplete === 'function') {
                onComplete();
            }
        } else {
            countdownElement.innerHTML = `Get ready: ${countdownSeconds}`;
        }
    }, 1000);
}

// Practice timer
function startPracticeTimer(seconds) {
    if (!practiceTimerElement) return;
    
    // If no-time option was selected (seconds = 0)
    if (seconds === 0) {
        practiceTimerElement.textContent = "No time limit";
        return;
    }
    
    let remainingSeconds = seconds;
    practiceTimerElement.textContent = formatTime(remainingSeconds);
    
    practiceTimer = setInterval(() => {
        remainingSeconds--;
        
        if (remainingSeconds <= 0) {
            clearInterval(practiceTimer);
            // Auto-redirect to feedback summary when timer ends
            endPracticeSession();
        } else {
            practiceTimerElement.textContent = formatTime(remainingSeconds);
        }
    }, 1000);
}

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
    if (!detector || video.readyState < 2 || !isAnalyzing) { 
        // Skip if detector not loaded, video not ready, or analysis stopped
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

            // Draw keypoints and skeleton (every frame for visual feedback)
            drawKeypoints(ctx, keypoints);
            drawSkeleton(ctx, keypoints);

            // --- TIMED FEEDBACK: Check time before giving posture feedback ---
            if (now - lastFeedbackTime >= feedbackInterval) {
                console.log("Generating feedback - interval passed.");
                
                // Generate appropriate feedback based on practice focus
                let postureFeedback;
                let hasPostureProblems = false;
                
                switch(practiceFocus) {
                    case 'head_neck':
                        postureFeedback = generateHeadNeckFeedback(keypoints);
                        // Check if feedback indicates a problem (doesn't contain "excellent" or "great")
                        hasPostureProblems = !postureFeedback.toLowerCase().includes("excellent") && !postureFeedback.toLowerCase().includes("great");
                        break;
                    case 'shoulders':
                        postureFeedback = generateShouldersFeedback(keypoints);
                        hasPostureProblems = !postureFeedback.toLowerCase().includes("excellent") && !postureFeedback.toLowerCase().includes("great");
                        break;
                    default:
                        postureFeedback = generateCompleteFeedback(keypoints);
                        hasPostureProblems = !postureFeedback.toLowerCase().includes("good") && !postureFeedback.toLowerCase().includes("great");
                }
                
                // Play audio alert if there are posture problems
                if (hasPostureProblems) {
                    playPostureAlert();
                }
                
                // Store feedback for summary
                const feedbackItem = {
                    timestamp: now,
                    message: postureFeedback,
                    keypoints: JSON.parse(JSON.stringify(keypoints)), // Deep copy keypoints
                };
                feedbackHistory.push(feedbackItem);
                
                // Update the UI
                feedback.innerHTML = postureFeedback;
                lastFeedbackTime = now; // Reset the timer
            }

        } else {
            // Optional: If no poses are detected after a while, update feedback
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
    }

    // Loop the detection
    animationFrameId = requestAnimationFrame(detectPose);
}

// Generate feedback ONLY for head and neck position
function generateHeadNeckFeedback(keypoints) {
    let feedbackMessages = [];
    
    // --- USING REDUCED TOLERANCE THRESHOLDS ---
    const neckCenteringThreshold = THRESHOLDS.neckCenteringThreshold;
    const headTiltThreshold = THRESHOLDS.headTiltThreshold;

    // Find keypoints needed for head/neck checks
    const leftShoulder = keypoints.find((kp) => kp.name === 'left_shoulder');
    const rightShoulder = keypoints.find((kp) => kp.name === 'right_shoulder');
    const leftEar = keypoints.find((kp) => kp.name === 'left_ear');
    const rightEar = keypoints.find((kp) => kp.name === 'right_ear');
    const nose = keypoints.find((kp) => kp.name === 'nose');

    // Check neck alignment (head centered over shoulders)
    if (nose && leftShoulder && rightShoulder && 
        nose.score > 0.5 && leftShoulder.score > 0.5 && rightShoulder.score > 0.5) {
        
        const neckCenter = (leftShoulder.x + rightShoulder.x) / 2;
        const neckOffset = Math.abs(nose.x - neckCenter);
        
        if (neckOffset > neckCenteringThreshold) {
            feedbackMessages.push("Center your head directly over your shoulders.");
        }
    }

    // Check head tilt (ears level)
    if (leftEar && rightEar && leftEar.score > 0.5 && rightEar.score > 0.5) {
        const earDiff = Math.abs(leftEar.y - rightEar.y);
        
        if (earDiff > headTiltThreshold) {
            feedbackMessages.push("Keep your head level, avoid tilting to either side.");
        }
    }
    
    // Check chin position using nose and neck points
    if (nose && leftShoulder && rightShoulder && 
        nose.score > 0.5 && leftShoulder.score > 0.5 && rightShoulder.score > 0.5) {
        
        // Calculate approximate neck position
        const neckY = (leftShoulder.y + rightShoulder.y) / 2;
        const neckX = (leftShoulder.x + rightShoulder.x) / 2;
        
        // Check if chin is tucked (nose should be slightly above and forward of neck)
        const idealNoseOffset = 25; // Approximate ideal offset value
        const actualOffset = nose.y - neckY;
        
        if (actualOffset > -idealNoseOffset) { // If nose is too low (chin not tucked)
            feedbackMessages.push("Gently tuck your chin toward your throat to lengthen the back of your neck.");
        } else if (actualOffset < -idealNoseOffset * 2) { // If chin is over-tucked
            feedbackMessages.push("Relax your chin slightly, avoid excessive tucking.");
        }
    }

    // --- Compile feedback ---
    if (feedbackMessages.length === 0) {
        if (leftShoulder && rightShoulder && leftEar && rightEar && nose) {
            return "Excellent head and neck position! Maintain this alignment.";
        } else {
            return "Position yourself clearly in the frame for head and neck analysis.";
        }
    } else {
        return feedbackMessages.join(" "); // Join multiple messages with a space
    }
}

// Generate feedback ONLY for shoulders position
function generateShouldersFeedback(keypoints) {
    let feedbackMessages = [];
    
    // --- THRESHOLDS ---
    const shoulderAlignmentThreshold = THRESHOLDS.shoulderAlignmentThreshold;
    const shoulderTensionThreshold = THRESHOLDS.shoulderTensionThreshold;
    
    // Find keypoints needed for shoulder checks
    const leftShoulder = keypoints.find((kp) => kp.name === 'left_shoulder');
    const rightShoulder = keypoints.find((kp) => kp.name === 'right_shoulder');
    const leftElbow = keypoints.find((kp) => kp.name === 'left_elbow');
    const rightElbow = keypoints.find((kp) => kp.name === 'right_elbow');
    const leftWrist = keypoints.find((kp) => kp.name === 'left_wrist');
    const rightWrist = keypoints.find((kp) => kp.name === 'right_wrist');
    
    // Check shoulder alignment (level shoulders)
    if (leftShoulder && rightShoulder && 
        leftShoulder.score > 0.5 && rightShoulder.score > 0.5) {
        
        const shoulderDiff = Math.abs(leftShoulder.y - rightShoulder.y);
        if (shoulderDiff > shoulderAlignmentThreshold) {
            feedbackMessages.push("Level your shoulders to be even with each other.");
        }
    }
    
    // Check shoulder tension (shoulders pulled up towards ears)
    if (leftShoulder && rightShoulder && leftElbow && rightElbow &&
        leftShoulder.score > 0.5 && rightShoulder.score > 0.5 && 
        leftElbow.score > 0.5 && rightElbow.score > 0.5) {
        
        // Calculate shoulder to elbow vertical distance
        const leftShoulderToElbow = Math.abs(leftShoulder.y - leftElbow.y);
        const rightShoulderToElbow = Math.abs(rightShoulder.y - rightElbow.y);
        
        // If shoulders are pulled up, this distance will be smaller
        if (leftShoulderToElbow < shoulderTensionThreshold || 
            rightShoulderToElbow < shoulderTensionThreshold) {
            feedbackMessages.push("Relax your shoulders down away from your ears.");
        }
    }
    
    // Check arm position for meditation posture
    if (leftShoulder && rightShoulder && leftElbow && rightElbow && leftWrist && rightWrist &&
        leftShoulder.score > 0.5 && rightShoulder.score > 0.5 && 
        leftElbow.score > 0.5 && rightElbow.score > 0.5 &&
        leftWrist.score > 0.5 && rightWrist.score > 0.5) {
        
        // Check if arms are too extended or too close
        const leftArmExtension = Math.sqrt(
            Math.pow(leftWrist.x - leftShoulder.x, 2) + 
            Math.pow(leftWrist.y - leftShoulder.y, 2)
        );
        
        const rightArmExtension = Math.sqrt(
            Math.pow(rightWrist.x - rightShoulder.x, 2) + 
            Math.pow(rightWrist.y - rightShoulder.y, 2)
        );
        
        // Threshold values would need adjustment based on frame size
        const maxExtension = 150; // Maximum comfortable extension
        const minExtension = 60;  // Minimum extension for relaxed position
        
        if (leftArmExtension > maxExtension || rightArmExtension > maxExtension) {
            feedbackMessages.push("Bring your hands closer to your body for a relaxed arm position.");
        } else if (leftArmExtension < minExtension || rightArmExtension < minExtension) {
            feedbackMessages.push("Allow some space between your arms and torso.");
        }
    }
    
    // --- Compile feedback ---
    if (feedbackMessages.length === 0) {
        if (leftShoulder && rightShoulder && leftElbow && rightElbow) {
            return "Great shoulder position! Keep them relaxed and balanced.";
        } else {
            return "Position your upper body clearly in the frame for shoulder analysis.";
        }
    } else {
        return feedbackMessages.join(" "); // Join multiple messages with a space
    }
}

// Generate comprehensive feedback for all posture elements
function generateCompleteFeedback(keypoints) {
    let feedbackMessages = [];
    
    // --- USING REDUCED TOLERANCE THRESHOLDS ---
    const shoulderAlignmentThreshold = THRESHOLDS.completeShoulderAlignmentThreshold;
    const neckCenteringThreshold = THRESHOLDS.completeNeckCenteringThreshold;
    const headTiltThreshold = THRESHOLDS.completeHeadTiltThreshold;

    // Find keypoints needed for upper body checks
    const leftShoulder = keypoints.find((kp) => kp.name === 'left_shoulder');
    const rightShoulder = keypoints.find((kp) => kp.name === 'right_shoulder');
    const leftEar = keypoints.find((kp) => kp.name === 'left_ear');
    const rightEar = keypoints.find((kp) => kp.name === 'right_ear');
    const nose = keypoints.find((kp) => kp.name === 'nose');

    // Check shoulder alignment
    if (leftShoulder && rightShoulder && leftShoulder.score > 0.5 && rightShoulder.score > 0.5) {
        const shoulderDiff = Math.abs(leftShoulder.y - rightShoulder.y);
        if (shoulderDiff > shoulderAlignmentThreshold) {
            feedbackMessages.push("Align shoulders evenly.");
        }
    }

    // Check neck alignment (head centered over shoulders)
    if (nose && leftShoulder && rightShoulder && nose.score > 0.5 && leftShoulder.score > 0.5 && rightShoulder.score > 0.5) {
        const neckCenter = (leftShoulder.x + rightShoulder.x) / 2;
        const neckOffset = Math.abs(nose.x - neckCenter);
        if (neckOffset > neckCenteringThreshold) {
            feedbackMessages.push("Center your head over your shoulders.");
        }
    }

    // Check head tilt (ears level)
    if (leftEar && rightEar && leftEar.score > 0.5 && rightEar.score > 0.5) {
        const earDiff = Math.abs(leftEar.y - rightEar.y);
        if (earDiff > headTiltThreshold) {
            feedbackMessages.push("Keep your head level (avoid tilting).");
        }
    }

    // --- Compile feedback ---
    if (feedbackMessages.length === 0) {
        if (leftShoulder && rightShoulder && leftEar && rightEar && nose) {
            return "Good posture! Keep it up.";
        } else {
            return "Keep still for analysis.";
        }
    } else {
        return feedbackMessages.join(" "); // Join multiple messages with a space
    }
}

// Rest of the existing functions (calculatePostureScores, generateFeedbackSummary, etc.) remain the same

// Main function to orchestrate setup and detection
async function startPosturePractice() {
    if (startButton) {
        startButton.disabled = true; // Disable button while setting up
        startButton.innerHTML = "Starting...";
    }
    
    // 1. Start countdown timer before activating camera
    startCountdown(5, async () => {
        // 2. Setup Camera after countdown
        const videoElement = await setupCamera();
        if (!videoElement) {
            if (startButton) {
                startButton.disabled = false; // Re-enable button if setup fails
                startButton.innerHTML = "Start Practice";
            }
            return; // Stop if camera failed
        }
        videoElement.play(); // Ensure video plays

        // 3. Load Model
        await loadModel(); // Wait for the model to load
        if (!detector) {
            if (startButton) {
                startButton.disabled = false; // Re-enable button if model fails
                startButton.innerHTML = "Start Practice";
            }
            feedback.innerHTML = "Failed to load analysis model. Please try again.";
            return; // Stop if model failed
        }

        // 4. Start Detection Loop
        lastFeedbackTime = Date.now() - feedbackInterval + 2000; // Give feedback soon after start
        console.log("Starting detection loop...");
        feedback.innerHTML = "Starting posture analysis...";
        
        // Set practice start flag
        practiceStartTime = Date.now();
        isAnalyzing = true;

        // Clear previous loop if any
        if (animationFrameId) {
            cancelAnimationFrame(animationFrameId);
        }
        
        // Show end practice button if it exists
        if (endPracticeButton) {
            endPracticeButton.style.display = 'inline-block';
        }
        
        // Start the practice timer
        startPracticeTimer(practiceSeconds);
        
        // Start the detection loop
        detectPose();
    });
}

// End the practice session and save results
function endPracticeSession() {
    // Stop analyzing
    isAnalyzing = false;
    
    // Stop timers
    if (practiceTimer) {
        clearInterval(practiceTimer);
    }
    
    // Stop camera stream if active
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    
    // Cancel animation frame if running
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    
    // Calculate final results
    const feedbackData = generateFeedbackSummary();
    
    // Show feedback message
    feedback.innerHTML = "Practice complete! Preparing your results...";
    
    // In a real implementation, this would be an AJAX call to save data to the server
    console.log("Practice session ended. Results:", feedbackData);
    
    // Redirect to feedback summary page
    window.location.href = "/feedback_summary";
}

// Cleanup resources when leaving the page
window.addEventListener('beforeunload', () => {
    // Stop camera if running
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
    }
    
    // Cancel animation frame if running
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
    }
    
    // Clear timers if running
    if (countdownTimer) {
        clearInterval(countdownTimer);
    }
    
    if (practiceTimer) {
        clearInterval(practiceTimer);
    }
});