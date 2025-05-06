// Verify TensorFlow.js is loaded
console.log('TensorFlow.js version:', tf.version.tfjs);

// Global variables
let lastFeedbackTime = 0;
const feedbackInterval = 15000; // 15 seconds in milliseconds (reduced from 30s)
let detector = null; // Hold the detector globally
let animationFrameId = null; // To control the animation loop
let countdownTimer = null; // For the countdown timer
let countdownSeconds = 5; // Default countdown seconds
let practiceDuration = 0; // Track practice duration in seconds
let practiceStartTime = null; // When practice actually started
let isAnalyzing = false; // Flag to track if analysis is running
let feedbackHistory = []; // Store feedback history for summary
let practiceFocus = 'all'; // Default to analyze all posture aspects

// Get elements needed multiple times
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas?.getContext('2d');
const feedback = document.getElementById('feedback');
const startButton = document.getElementById('start-button');
const countdownElement = document.getElementById('countdown');
const endPracticeButton = document.getElementById('end-practice-button');

// Initialize based on page focus (if set)
document.addEventListener('DOMContentLoaded', function() {
    // Check if practice focus is set on the page
    const focusElement = document.getElementById('practice-focus');
    if (focusElement) {
        practiceFocus = focusElement.getAttribute('data-focus') || 'all';
        console.log(`Practice focus set to: ${practiceFocus}`);
    }
    
    // Setup event listeners
    if (startButton) {
        startButton.addEventListener('click', startPosturePractice);
    }
    
    if (endPracticeButton) {
        endPracticeButton.addEventListener('click', endPracticeSession);
    }
});

// Countdown timer function
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

        // Update practice duration
        if (practiceStartTime) {
            practiceDuration = Math.floor((now - practiceStartTime) / 1000);
            
            // Update duration display if element exists
            const durationElement = document.getElementById('practice-duration');
            if (durationElement) {
                const minutes = Math.floor(practiceDuration / 60);
                const seconds = practiceDuration % 60;
                durationElement.textContent = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            }
        }

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
                
                switch(practiceFocus) {
                    case 'head_neck':
                        postureFeedback = generateHeadNeckFeedback(keypoints);
                        break;
                    case 'shoulders':
                        postureFeedback = generateShouldersFeedback(keypoints);
                        break;
                    case 'spine':
                        postureFeedback = generateSpineFeedback(keypoints);
                        break;
                    default:
                        postureFeedback = generateCompleteFeedback(keypoints);
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

// Generate feedback ONLY for head and neck position
function generateHeadNeckFeedback(keypoints) {
    let feedbackMessages = [];
    
    // --- INCREASED TOLERANCE THRESHOLDS ---
    const neckCenteringThreshold = 30;   // Reduced tolerance for focused feedback
    const headTiltThreshold = 25;       // Reduced tolerance for focused feedback

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
    const shoulderAlignmentThreshold = 30; // Reduced threshold for focused feedback
    const shoulderTensionThreshold = 40;   // New threshold for shoulder elevation
    
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

// Generate feedback for spine position (approximated from available keypoints)
function generateSpineFeedback(keypoints) {
    let feedbackMessages = [];
    
    // Find keypoints needed for spine approximation
    const leftShoulder = keypoints.find((kp) => kp.name === 'left_shoulder');
    const rightShoulder = keypoints.find((kp) => kp.name === 'right_shoulder');
    const leftHip = keypoints.find((kp) => kp.name === 'left_hip');
    const rightHip = keypoints.find((kp) => kp.name === 'right_hip');
    
    // Check if we have the minimum points needed
    if (leftShoulder && rightShoulder && leftHip && rightHip &&
        leftShoulder.score > 0.5 && rightShoulder.score > 0.5 &&
        leftHip.score > 0.5 && rightHip.score > 0.5) {
        
        // Calculate midpoints for shoulders and hips
        const shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2;
        const shoulderMidY = (leftShoulder.y + rightShoulder.y) / 2;
        const hipMidX = (leftHip.x + rightHip.x) / 2;
        const hipMidY = (leftHip.y + rightHip.y) / 2;
        
        // Calculate vertical alignment
        const verticalOffset = Math.abs(shoulderMidX - hipMidX);
        const verticalThreshold = 25; // Threshold for vertical alignment
        
        if (verticalOffset > verticalThreshold) {
            feedbackMessages.push("Align your spine vertically by stacking shoulders directly over hips.");
        }
        
        // Calculate torso angle to detect slouching or over-arching
        const torsoAngle = Math.atan2(shoulderMidY - hipMidY, shoulderMidX - hipMidX) * 180 / Math.PI;
        const idealAngle = 90; // Perfectly vertical
        const angleThreshold = 10; // Threshold for deviation
        
        const angleDiff = Math.abs(torsoAngle - idealAngle);
        if (angleDiff > angleThreshold) {
            if (torsoAngle < idealAngle) {
                feedbackMessages.push("Avoid slouching forward. Sit taller through your spine.");
            } else {
                feedbackMessages.push("Avoid over-arching your back. Find a neutral spine position.");
            }
        }
    } else {
        return "Position yourself to show both shoulders and hips for spine analysis.";
    }
    
    // --- Compile feedback ---
    if (feedbackMessages.length === 0) {
        return "Good spine alignment! Maintain this natural position.";
    } else {
        return feedbackMessages.join(" "); // Join multiple messages with a space
    }
}

// Generate comprehensive feedback for all posture elements
function generateCompleteFeedback(keypoints) {
    let feedbackMessages = [];
    
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
    const leftHip = keypoints.find((kp) => kp.name === 'left_hip');
    const rightHip = keypoints.find((kp) => kp.name === 'right_hip');

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
    
    // Check spine alignment if possible
    if (leftShoulder && rightShoulder && leftHip && rightHip &&
        leftShoulder.score > 0.5 && rightShoulder.score > 0.5 &&
        leftHip.score > 0.5 && rightHip.score > 0.5) {
        
        // Calculate midpoints for shoulders and hips
        const shoulderMidX = (leftShoulder.x + rightShoulder.x) / 2;
        const hipMidX = (leftHip.x + rightHip.x) / 2;
        
        // Calculate vertical alignment
        const verticalOffset = Math.abs(shoulderMidX - hipMidX);
        const verticalThreshold = 35; // Slightly higher threshold for general practice
        
        if (verticalOffset > verticalThreshold) {
            feedbackMessages.push("Align your spine vertically.");
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

// Calculate posture scores based on feedback history
function calculatePostureScores() {
    // Define initial scores
    const scores = {
        head_neck_score: 0,
        shoulders_score: 0,
        spine_score: 0,
        overall_score: 0
    };
    
    // Count total feedback instances and "good" feedback instances
    let totalHeadNeckFeedback = 0;
    let goodHeadNeckFeedback = 0;
    let totalShouldersFeedback = 0;
    let goodShouldersFeedback = 0;
    let totalSpineFeedback = 0;
    let goodSpineFeedback = 0;
    
    feedbackHistory.forEach(item => {
        // Check if feedback included head/neck issues
        if (item.message.includes("head") || item.message.includes("neck") || 
            item.message.includes("chin") || item.message.includes("tilt")) {
            totalHeadNeckFeedback++;
            if (item.message.includes("Excellent head") || item.message.includes("Good") || 
                item.message.includes("Keep it up")) {
                goodHeadNeckFeedback++;
            }
        }
        
        // Check if feedback included shoulder issues
        if (item.message.includes("shoulder") || item.message.includes("arm") || 
            item.message.includes("chest")) {
            totalShouldersFeedback++;
            if (item.message.includes("Great shoulder") || item.message.includes("Good") || 
                item.message.includes("Keep it up")) {
                goodShouldersFeedback++;
            }
        }
        
        // Check if feedback included spine issues
        if (item.message.includes("spine") || item.message.includes("back") || 
            item.message.includes("slouch") || item.message.includes("tall")) {
            totalSpineFeedback++;
            if (item.message.includes("Good spine") || item.message.includes("Good") || 
                item.message.includes("Keep it up")) {
                goodSpineFeedback++;
            }
        }
    });
    
    // Calculate scores as percentages
    if (totalHeadNeckFeedback > 0) {
        scores.head_neck_score = Math.round((goodHeadNeckFeedback / totalHeadNeckFeedback) * 100);
    } else {
        // Default score if no feedback data available
        scores.head_neck_score = 50;
    }
    
    if (totalShouldersFeedback > 0) {
        scores.shoulders_score = Math.round((goodShouldersFeedback / totalShouldersFeedback) * 100);
    } else {
        scores.shoulders_score = 50;
    }
    
    if (totalSpineFeedback > 0) {
        scores.spine_score = Math.round((goodSpineFeedback / totalSpineFeedback) * 100);
    } else {
        scores.spine_score = 50;
    }
    
    // Calculate overall score (weighted average)
    scores.overall_score = Math.round(
        (scores.head_neck_score + scores.shoulders_score + scores.spine_score) / 3
    );
    
    return scores;
}

// Generate feedback summary for the session
function generateFeedbackSummary() {
    const scores = calculatePostureScores();
    
    // Format practice duration
    const minutes = Math.floor(practiceDuration / 60);
    const seconds = practiceDuration % 60;
    const durationFormatted = `${minutes}:${seconds.toString().padStart(2, '0')}`;
    
    // Create feedback objects
    let feedbackData = {
        head_neck_score: scores.head_neck_score,
        shoulders_score: scores.shoulders_score,
        spine_score: scores.spine_score,
        overall_score: scores.overall_score,
        practice_duration: durationFormatted,
        head_neck_feedback: generateHeadNeckSummary(scores.head_neck_score),
        shoulders_feedback: generateShouldersSummary(scores.shoulders_score),
        spine_feedback: generateSpineSummary(scores.spine_score),
        overall_feedback: generateOverallSummary(scores.overall_score)
    };
    
    // Add detailed metrics for results page
    feedbackData = {
        ...feedbackData,
        head_alignment: getHeadAlignmentRating(scores.head_neck_score),
        head_alignment_score: Math.min(scores.head_neck_score + 10, 100),
        neck_tension: getNeckTensionRating(scores.head_neck_score),
        neck_tension_score: Math.max(scores.head_neck_score - 5, 0),
        gaze_direction: getGazeDirectionRating(scores.head_neck_score),
        gaze_direction_score: scores.head_neck_score,
        shoulder_level: getShoulderLevelRating(scores.shoulders_score),
        shoulder_level_score: scores.shoulders_score,
        shoulder_tension: getShoulderTensionRating(scores.shoulders_score),
        shoulder_tension_score: Math.max(scores.shoulders_score - 10, 0),
        chest_openness: getChestOpennessRating(scores.shoulders_score),
        chest_openness_score: Math.min(scores.shoulders_score + 5, 100),
        spine_curvature: getSpineCurvatureRating(scores.spine_score),
        spine_curvature_score: scores.spine_score,
        lower_back_support: getLowerBackSupportRating(scores.spine_score),
        lower_back_support_score: Math.max(scores.spine_score - 15, 0),
        stability: getStabilityRating(scores.spine_score),
        stability_score: Math.min(scores.spine_score + 10, 100)
    };
    
    return feedbackData;
}

// Helper functions for generating specific ratings
function getHeadAlignmentRating(score) {
    if (score >= 80) return "Excellent";
    if (score >= 60) return "Good";
    return "Needs Work";
}

function getNeckTensionRating(score) {
    if (score >= 80) return "Relaxed";
    if (score >= 60) return "Slightly Tense";
    return "Tense";
}

function getGazeDirectionRating(score) {
    if (score >= 80) return "Level";
    if (score >= 60) return "Slightly Off";
    return "Misaligned";
}

function getShoulderLevelRating(score) {
    if (score >= 80) return "Even";
    if (score >= 60) return "Slightly Uneven";
    return "Uneven";
}

function getShoulderTensionRating(score) {
    if (score >= 80) return "Relaxed";
    if (score >= 60) return "Slightly Tense";
    return "Tense";
}

function getChestOpennessRating(score) {
    if (score >= 80) return "Open";
    if (score >= 60) return "Partially Open";
    return "Closed";
}

function getSpineCurvatureRating(score) {
    if (score >= 80) return "Natural";
    if (score >= 60) return "Slight Deviation";
    return "Unnatural";
}

function getLowerBackSupportRating(score) {
    if (score >= 80) return "Well Supported";
    if (score >= 60) return "Moderate Support";
    return "Needs Support";
}

function getStabilityRating(score) {
    if (score >= 80) return "Stable";
    if (score >= 60) return "Moderately Stable";
    return "Unstable";
}

// Generate text summaries based on scores
function generateHeadNeckSummary(score) {
    if (score >= 80) {
        return "Your head and neck alignment is excellent. You maintain good posture with your head properly centered over your shoulders.";
    } else if (score >= 60) {
        return "Your head and neck position is good, but could use some improvement. Try to keep your chin slightly tucked and ears level.";
    } else {
        return "Your head and neck positioning needs work. Focus on centering your head over your shoulders and keeping your gaze level.";
    }
}

function generateShouldersSummary(score) {
    if (score >= 80) {
        return "Your shoulders are well-positioned - relaxed and level. You maintain good shoulder posture throughout your practice.";
    } else if (score >= 60) {
        return "Your shoulder position is good but could be improved. Remember to keep shoulders down away from your ears and evenly balanced.";
    } else {
        return "Your shoulders show tension and uneven positioning. Practice relaxing them down and back, keeping them level with each other.";
    }
}

function generateSpineSummary(score) {
    if (score >= 80) {
        return "Your spine maintains its natural curves and good alignment. You sit tall with good core engagement.";
    } else if (score >= 60) {
        return "Your spine alignment is adequate but shows some room for improvement. Focus on maintaining a tall, natural position.";
    } else {
        return "Your spine alignment needs work. Practice sitting taller and maintaining the natural curves of your spine without slouching or over-arching.";
    }
}

function generateOverallSummary(score) {
    if (score >= 80) {
        return "Your overall posture is excellent! You maintain good alignment throughout your practice, which will help prevent discomfort and injury.";
    } else if (score >= 60) {
        return "Your overall posture is good with some areas that could use improvement. With regular practice, you'll develop better postural habits.";
    } else {
        return "Your posture needs consistent work to improve alignment. Focus on the specific recommendations for each body area to make progress.";
    }
}

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
        
        // Set practice start time
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
        
        // Start the detection loop
        detectPose();
    });
}

// End the practice session and save results
function endPracticeSession() {
    // Stop analyzing
    isAnalyzing = false;
    
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
    
    // Store in session (will need backend implementation)
    console.log("Practice session ended. Results:", feedbackData);
    
    // Show feedback message
    feedback.innerHTML = "Practice complete! Preparing your results...";
    
    // Redirect to summary page (this would need to be implemented on the backend)
    // For now, we'll just log the data that would be sent
    console.log("Would redirect to feedback summary with data:", feedbackData);
    
    // In a real implementation, you would:
    // 1. Send the data to the server (e.g., via fetch/AJAX)
    // 2. Store it in the session on the server
    // 3. Redirect to the summary page
    
    // Simulate redirect (in a real app, this would happen after the data is saved)
    setTimeout(() => {
        // This would be replaced with a real redirect:
        window.location.href = "/feedback_summary";
    }, 2000);
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
    
    // Clear countdown timer if running
    if (countdownTimer) {
        clearInterval(countdownTimer);
    }
});

// If startButton exists, set up the click event
if (startButton) {
    startButton.addEventListener('click', startPosturePractice);
}