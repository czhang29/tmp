// Timer Selection and Countdown Functionality

document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the timer selection page
    const timerSelectionContainer = document.getElementById('timer-selection');
    if (timerSelectionContainer) {
        console.log("Timer selection page detected");
        setupTimerSelection();
    }

    // Check if we're on the countdown page
    const countdownElement = document.getElementById('fullscreen-countdown');
    if (countdownElement) {
        console.log("Countdown page detected");
        startFullscreenCountdown();
    }

    // Setup audio toggle if exists
    const audioToggle = document.getElementById('audio-toggle');
    if (audioToggle) {
        audioToggle.addEventListener('click', toggleAudio);
    }
});

// Function to set up timer selection
function setupTimerSelection() {
    const timerOptions = document.querySelectorAll('.timer-option');
    const continueButton = document.getElementById('continue-button');
    const selectedTimerInput = document.getElementById('selected-timer');
    
    if (!timerOptions || !continueButton || !selectedTimerInput) {
        console.error("Could not find timer elements");
        return;
    }
    
    // Default selection is already set in HTML
    
    timerOptions.forEach(option => {
        option.addEventListener('click', function() {
            const seconds = this.getAttribute('data-seconds');
            console.log("Selected timer:", seconds);
            
            // Update visual selection
            timerOptions.forEach(opt => opt.classList.remove('selected'));
            this.classList.add('selected');
            
            // Update hidden input
            selectedTimerInput.value = seconds;
        });
    });
}

// Function to handle fullscreen countdown
function startFullscreenCountdown() {
    const countdownNumber = document.getElementById('countdown-number');
    const redirectUrlInput = document.getElementById('redirect-url');
    
    if (!countdownNumber || !redirectUrlInput) {
        console.error("Could not find countdown elements");
        return;
    }
    
    const redirectUrl = redirectUrlInput.value;
    let seconds = 5;
    
    // Initial display
    countdownNumber.textContent = seconds;
    
    // Start countdown
    const countdownInterval = setInterval(() => {
        seconds--;
        
        if (seconds <= 0) {
            clearInterval(countdownInterval);
            window.location.href = redirectUrl;
        } else {
            countdownNumber.textContent = seconds;
        }
    }, 1000);
}

// Audio toggle function
function toggleAudio() {
    const audioIcon = this.querySelector('i');
    
    if (this.classList.contains('muted')) {
        this.classList.remove('muted');
        audioIcon.className = 'fas fa-volume-up';
        window.audioEnabled = true;
    } else {
        this.classList.add('muted');
        audioIcon.className = 'fas fa-volume-mute';
        window.audioEnabled = false;
    }
    
    console.log("Audio enabled:", window.audioEnabled);
}