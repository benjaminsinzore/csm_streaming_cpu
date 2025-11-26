// chat.js

// --- State Variables ---
let ws;
let sessionStartTime = null;
let messageCount = 0;
let audioLevelsChart = null;
let isRecording = false;
let isAudioCurrentlyPlaying = false;
let configSaved = false;
let currentAudioSource = null;
let interruptRequested = false;
let interruptInProgress = false;
let audioContext = null;
let lastSeenGenId = 0;
let reconnecting = false;
let reconnectAttempts = 0;
let maxReconnectAttempts = 10;
const SESSION_ID = "default_" + Date.now();
console.log("chat.js loaded - Session ID:", SESSION_ID);

let micStream;
let selectedMicId = null;
let selectedOutputId = null;
let audioPlaybackQueue = []; // Queue for audio chunks
let activeGenId = 0; // Track the current generation ID for audio playback

// --- DOM Ready and Initialization ---
document.addEventListener('DOMContentLoaded', function () {
    console.log("🚀 DOM loaded, initializing chat...");
    loadUserInfo();
    loadConversationHistory(); // Load history on page load
    connectWebSocket(); // Connect to WebSocket
    initializeAudioChart(); // Initialize audio visualization chart
    setupEventListeners(); // Setup button and input listeners
    checkSystemStatus(); // Check backend status
    startDurationTimer(); // Start session duration timer
    initializeSettingsModal(); // Setup settings modal interactions

    // Make debug functions available globally for console access
    window.debugConversationHistory = debugConversationHistory;
    window.debugWebSocket = debugWebSocket;
    window.testMessageDisplay = testMessageDisplay;
    window.testFullFlow = testFullFlow;
    window.checkConversationStyles = checkConversationStyles;
});

function setupEventListeners() {
    console.log("🔧 Setting up event listeners...");
    const sendBtn = document.getElementById('sendTextBtn');
    const textInput = document.getElementById('textInput');
    const micBtn = document.getElementById('micToggleBtn');
    const interruptBtn = document.getElementById('interruptBtn');
    const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');
    const settingsBtn = document.getElementById('settingsBtn');
    const closeSettingsBtn = document.getElementById('closeSettingsBtn');

    if (sendBtn) {
        sendBtn.onclick = () => {
            try {
                sendTextMessage(textInput.value);
            } catch (error) {
                console.error("Error in send button handler:", error);
            }
        };
    }

    if (textInput) {
        textInput.addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                try {
                    sendTextMessage(textInput.value);
                } catch (error) {
                    console.error("Error in text input handler:", error);
                }
            } else if (e.key === 'Escape') {
                e.preventDefault();
                requestInterrupt();
            }
        });
    }

    if (micBtn) {
        micBtn.addEventListener('click', toggleMic);
    }

    if (interruptBtn) {
        interruptBtn.addEventListener('click', requestInterrupt);
    }

    if (refreshHistoryBtn) {
        // Corrected: Use 'this.loadConversationHistory()' or just 'loadConversationHistory()'
        // 'this' here refers to the button element, not the global scope where the function is defined
        refreshHistoryBtn.addEventListener('click', () => {
            loadConversationHistory(); // Fetch and display the latest history
        });
    }

    if (settingsBtn) {
        settingsBtn.addEventListener('click', () => {
            document.getElementById('settingsModal').classList.remove('hidden');
        });
    }

    if (closeSettingsBtn) {
        closeSettingsBtn.addEventListener('click', () => {
            document.getElementById('settingsModal').classList.add('hidden');
        });
    }

    // Add debug button
    const btn = document.getElementById('sendTextBtn');
    if (btn && btn.parentElement) {
        const debugBtn = document.createElement('button');
        debugBtn.innerText = "Debug Audio";
        debugBtn.className = "px-3 py-2 ml-2 bg-blue-600 text-white rounded text-xs";
        debugBtn.onclick = () => {
            console.log("- Debug info:");
            console.log("- Audio playing:", isAudioCurrentlyPlaying);
            console.log("- Interrupt requested:", interruptRequested);
            console.log("- Interrupt in progress:", interruptInProgress);
            console.log("- Current source:", currentAudioSource);
            console.log("- Queue length:", audioPlaybackQueue.length);
            console.log("- Audio context state:", audioContext?.state);
            console.log("- Active generation ID:", activeGenId);
            console.log("- Last seen generation ID:", lastSeenGenId);
            console.log("- WebSocket state:", ws ? ws.readyState : "no websocket");
            showNotification("Debug info in console", "info");
        };
        btn.parentElement.appendChild(debugBtn);
    }

    // Run the update function periodically for interrupt button state
    setInterval(() => {
        const interruptBtn = document.getElementById('interruptBtn');
        if (interruptBtn) {
            if (isAudioCurrentlyPlaying && !interruptRequested && !interruptInProgress) {
                interruptBtn.disabled = false;
                interruptBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            } else {
                interruptBtn.disabled = true;
                interruptBtn.classList.add('opacity-50', 'cursor-not-allowed');
            }
        }
    }, 300);

    // Add keyboard shortcut for interrupt (Space or Escape)
    document.addEventListener('keydown', (e) => {
        if ((e.code === 'Space' || e.code === 'Escape') && (isAudioCurrentlyPlaying || interruptInProgress)) {
            e.preventDefault(); // Prevent default space scroll behavior
            requestInterrupt();
        }
    });

    console.log("✅ Event listeners set up.");
}

// --- WebSocket Connection ---
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws?session_id=${SESSION_ID}`;
    console.log("🔌 Connecting to WebSocket:", wsUrl);

    ws = new WebSocket(wsUrl);

    ws.onopen = (event) => {
        console.log("✅ WebSocket connected to:", wsUrl);
        const connLbl = document.getElementById('connectionStatus');
        if (connLbl) {
            connLbl.textContent = 'Connected';
            connLbl.className = 'text-green-400';
        }
        reconnectAttempts = 0; // Reset attempts on successful connect
        reconnecting = false;
    };

    ws.onclose = (event) => {
        console.log("🔌 WebSocket disconnected. Code:", event.code, "Reason:", event.reason);
        const connLbl = document.getElementById('connectionStatus');
        if (connLbl) {
            connLbl.textContent = 'Disconnected';
            connLbl.className = 'text-red-500';
        }
        clearAudioPlayback(); // Stop any audio on disconnect
        // Attempt to reconnect if not a clean close (1000) or going away (1001)
        if (event.code !== 1000 && event.code !== 1001) {
            if (!reconnecting && reconnectAttempts < maxReconnectAttempts) {
                reconnecting = true;
                reconnectAttempts++;
                const delay = Math.min(1000 * Math.pow(1.5, reconnectAttempts), 10000); // Max 10 seconds
                console.log(`🔄 Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${maxReconnectAttempts})`);
                setTimeout(connectWebSocket, delay);
            } else if (reconnectAttempts >= maxReconnectAttempts) {
                console.error("❌ Max reconnection attempts reached. Giving up.");
                showNotification("Could not reconnect to server. Please refresh the page.", "error");
            }
        }
    };

    ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
        const connLbl = document.getElementById('connectionStatus');
        if (connLbl) {
            connLbl.textContent = 'Error';
            connLbl.className = 'text-red-500';
        }
    };

    ws.onmessage = (e) => {
        console.log("📨 RAW WebSocket message received:", e.data);
        try {
            const data = JSON.parse(e.data);
            console.log("📦 Parsed message:", data);

            switch (data.type) {
                case 'text':
                    console.log("📝 Processing user text message...");
                    addMessageToConversation('user', data.text);
                    showVoiceCircle();
                    break;
                case 'transcription':
                    console.log("🎤 Processing transcription...");
                    addMessageToConversation('user', data.text);
                    showVoiceCircle();
                    break;
                case 'response':
                    console.log("🤖 Processing AI response...");
                    addMessageToConversation('ai', data.text);
                    showVoiceCircle();
                    // Refresh history when new response is received
                    setTimeout(() => loadConversationHistory(), 1000); // Delay to allow DB write
                    break;
                case 'audio_chunk':
                    console.log("🔊 Audio chunk received - genId:", data.gen_id, "activeGenId:", activeGenId);
                    if (activeGenId === 0 && data.gen_id) {
                        activeGenId = data.gen_id;
                        console.log("🎯 Setting active generation ID to:", activeGenId);
                    }
                    if (data.gen_id === activeGenId) {
                        console.log("🎵 Queueing audio chunk for ID:", data.gen_id);
                        queueAudioChunk(data.audio_data);
                    } else {
                        console.log("⏭️ Received chunk for different generation ID, ignoring.");
                    }
                    break;
                case 'audio_generation_status':
                    console.log("🎵 Audio generation status:", data.status);
                    if (data.status === 'generating') {
                        console.log("🔄 New audio generation starting");
                        interruptRequested = false;
                        interruptInProgress = false;
                        if (data.gen_id) {
                            activeGenId = data.gen_id;
                            console.log("🎯 Active generation set to:", activeGenId);
                        }
                        showVoiceCircle();
                    } else if (data.status === 'first_chunk') {
                        console.log("🎵 First audio chunk ready");
                        showVoiceCircle();
                    } else if (data.status === 'complete') {
                        console.log("✅ Audio generation complete");
                        activeGenId = 0; // Reset active ID only after complete
                        if (!isAudioCurrentlyPlaying) {
                            hideVoiceCircle();
                        }
                    } else if (data.status === 'interrupted' || data.status === 'interrupt_acknowledged') {
                        console.log("⏹️ Audio interrupted by server");
                        clearAudioPlayback();
                    }
                    break;
                case 'status':
                    console.log("ℹ️ Status:", data.message);
                    if (data.message === 'Thinking...') {
                        showVoiceCircle();
                    }
                    break;
                case 'error':
                    console.error("❌ Error:", data.message);
                    showNotification(data.message, 'error');
                    hideVoiceCircle();
                    break;
                case 'vad_status':
                    console.log("🎤 VAD Status:", data.status);
                    if (data.status === 'speech_started') {
                        showVoiceCircle();
                    }
                    break;
                case 'test_response':
                    console.log("✅ Test response received:", data.message);
                    showNotification(data.message, 'success');
                    break;
                default:
                    console.log("❓ Unknown message type:", data.type);
            }
        } catch (error) {
            console.error("❌ Error parsing WebSocket message:", error, "Raw:", e.data);
        }
    };
}

// --- Message Handling ---
function addMessageToConversation(sender, text, timestamp = new Date().toLocaleTimeString()) {
    console.log(`💬 Adding ${sender} message:`, text.substring(0, 50) + (text.length > 50 ? '...' : ''));
    const pane = document.getElementById('conversationHistory');
    if (!pane) {
        console.error("❌ Conversation history pane not found! Check if element exists in HTML.");
        // Try to find the element with different selectors
        const alternativeSelectors = ['#conversationHistory', '.conversation-container div', '[id*="conversation"]', '[class*="conversation"]'];
        for (const selector of alternativeSelectors) {
            const element = document.querySelector(selector);
            if (element) {
                console.log(`Found conversation pane with selector: ${selector}`);
                // Retry adding the message to the found element
                addMessageToConversation(sender, text, timestamp);
                return; // Exit after retrying
            }
        }
        console.error("❌ Could not find conversation history pane with common selectors.");
        return; // Stop if pane is still not found
    }

    // Escape HTML to prevent injection
    const escapeHtml = (unsafe) => {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "<")
            .replace(/>/g, ">")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    };
    const escapedText = escapeHtml(text);

    const box = document.createElement('div');
    box.className = 'mb-4 p-4 bg-gray-800 rounded-lg shadow'; // Consistent styling
    box.innerHTML = `
        <div class="flex items-center">
            <div class="w-6 h-6 rounded-full flex items-center justify-center ${sender === 'user' ? 'bg-gray-300 text-gray-800' : 'bg-indigo-500 text-white'}">
                ${sender === 'user' ? 'U' : 'AI'}
            </div>
            <span class="text-xs text-gray-400 ml-2">${timestamp}</span>
        </div>
        <div class="text-white mt-1 text-sm">${escapedText}</div>
    `;
    console.log("✅ Message element created, appending to pane...");
    pane.appendChild(box);

    // Force multiple scroll methods to ensure visibility
    pane.scrollTop = pane.scrollHeight;
    setTimeout(() => {
        pane.scrollTop = pane.scrollHeight;
    }, 50);

    console.log("✅ User message should be visible. Total messages in pane:", pane.children.length);

    // Update message count
    const cnt = document.getElementById('messageCount');
    if (cnt) {
        messageCount++;
        cnt.textContent = messageCount;
        console.log("✅ Message count updated to:", messageCount);
    }

    // Clear input
    const textInput = document.getElementById('textInput');
    if (textInput) {
        textInput.value = '';
        console.log("✅ Input cleared");
    }

    // Show thinking indicator
    showVoiceCircle();
}

// --- Audio Handling ---
function queueAudioChunk(base64AudioData) {
    audioPlaybackQueue.push(base64AudioData);
    console.log("🎵 Queued audio chunk. Queue length:", audioPlaybackQueue.length);
    if (!isAudioCurrentlyPlaying) {
        processAudioQueue();
    }
}

async function processAudioQueue() {
    if (audioPlaybackQueue.length === 0) {
        console.log("🔚 Audio queue is empty, stopping playback loop.");
        isAudioCurrentlyPlaying = false;
        // Potentially hide the voice circle here if no more audio is expected for this generation
        // if (activeGenId === 0) { hideVoiceCircle(); } // Or check if generation is complete via status message
        return;
    }

    if (interruptRequested || interruptInProgress) {
        console.log("🛑 Interrupt requested, clearing queue and stopping playback.");
        audioPlaybackQueue = []; // Clear the queue
        return; // Exit without playing anything further
    }

    isAudioCurrentlyPlaying = true;
    const audioData = audioPlaybackQueue.shift(); // Get the next chunk

    try {
        // Decode base64 to binary
        const binaryString = atob(audioData);
        const len = binaryString.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }

        // Create audio buffer
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            window.audioContext = audioContext;
        }
        const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);

        // Create source and connect to destination
        currentAudioSource = audioContext.createBufferSource();
        currentAudioSource.buffer = audioBuffer;
        currentAudioSource.connect(audioContext.destination);

        // Play the audio
        console.log("▶️ Playing audio chunk...");
        currentAudioSource.start();

        // Set up end event to process next chunk
        currentAudioSource.onended = () => {
            console.log("⏹️ Audio chunk finished playing.");
            currentAudioSource = null; // Clear reference after playback
            processAudioQueue(); // Process the next chunk in the queue
        };

    } catch (error) {
        console.error("❌ Error processing audio chunk:", error);
        // Continue with the next chunk even if one fails
        processAudioQueue();
    }
}

function clearAudioPlayback() {
    console.log("🧹 Clearing audio playback...");
    interruptRequested = false; // Reset interrupt flag when clearing
    interruptInProgress = true; // Set in-progress flag to prevent new plays

    // Stop the current audio source
    try {
        if (currentAudioSource) {
            if (currentAudioSource.stop) {
                currentAudioSource.stop(0); // Stop immediately
            }
            currentAudioSource.disconnect(); // Disconnect from context
        }
    } catch (e) {
        console.warn("Error disconnecting/stopping audio source:", e);
    }
    currentAudioSource = null; // Clear the reference

    // Recreate the audio context to ensure complete stop (handles overlapping plays)
    try {
        if (audioContext) {
            const oldContext = audioContext;
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            window.audioContext = audioContext; // Update global reference
            try {
                oldContext.close(); // Close the old context to free resources
            } catch (closeError) {
                console.warn("Error closing old audio context:", closeError);
            }
        } else {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
            window.audioContext = audioContext;
        }
    } catch (contextError) {
        console.error("Error recreating audio context:", contextError);
    }

    // Clear the playback queue
    audioPlaybackQueue = [];
    isAudioCurrentlyPlaying = false;
    hideVoiceCircle();
    console.log("Audio playback cleared successfully");

    // Reset the in-progress flag after a short delay to allow for cleanup
    setTimeout(() => {
        interruptInProgress = false;
    }, 300);
}

function requestInterrupt() {
    console.log("User requested interruption");
    if (interruptInProgress) {
        console.log("Interrupt already in progress - force clearing again");
        clearAudioPlayback();
        return false; // Prevent multiple interrupt requests while processing
    }
    interruptRequested = true;
    interruptInProgress = true;
    showNotification("Interrupting...", "info");

    // Update button visual state
    const interruptBtn = document.getElementById('interruptBtn');
    if (interruptBtn) {
        interruptBtn.classList.add('bg-red-800');
        setTimeout(() => {
            interruptBtn.classList.remove('bg-red-800');
        }, 300);
    }

    // Send interrupt request to the server
    if (ws && ws.readyState === WebSocket.OPEN) {
        console.log("Sending interrupt request to server");
        try {
            ws.send(JSON.stringify({
                type: 'interrupt',
                immediate: true,
                session_id: SESSION_ID
            }));
        } catch (error) {
            console.error("Error sending interrupt request:", error);
        }
        // Timeout to ensure client-side cleanup if server doesn't respond quickly
        setTimeout(() => {
            if (interruptInProgress) {
                console.log("Interrupt timeout reached, forcing clear.");
                clearAudioPlayback();
            }
        }, 1000);
    } else {
        console.warn("WebSocket not open, clearing audio client-side only.");
        clearAudioPlayback();
    }
    return true;
}

// --- Voice Activity Detection (VAD) and Microphone ---
let vadProcessor = null;
let isVADActive = true; // Default to enabled
let vadThreshold = 0.5; // Default threshold
let volumeLevel = 1.0; // Default volume multiplier

async function toggleMic() {
    const micBtn = document.getElementById('micToggleBtn');
    const micStatus = document.getElementById('micStatus');
    const audioStatus = document.getElementById('audioStatus');

    if (isRecording) {
        console.log("⏹️ Stopping microphone...");
        isRecording = false;
        if (micBtn) {
            micBtn.classList.remove('pulse', 'bg-red-600');
            micBtn.classList.add('bg-indigo-600');
        }
        if (micStatus) micStatus.textContent = "Click to speak";
        if (audioStatus) audioStatus.textContent = "Idle";
        hideVoiceCircle();

        if (micStream) {
            micStream.getTracks().forEach(track => track.stop());
            micStream = null;
        }
        if (vadProcessor) {
            vadProcessor.stop();
            vadProcessor = null;
        }
        // Send an empty audio chunk to signal end of speech to the backend
        if (ws && ws.readyState === WebSocket.OPEN) {
            try {
                ws.send(JSON.stringify({
                    type: 'audio_chunk',
                    audio_data: '', // Empty chunk signals end
                    session_id: SESSION_ID
                }));
            } catch (error) {
                console.error("Error sending end-of-speech signal:", error);
            }
        }
    } else {
        console.log("🎤 Starting microphone...");
        isRecording = true;
        if (micBtn) {
            micBtn.classList.add('pulse', 'bg-red-600');
            micBtn.classList.remove('bg-indigo-600');
        }
        if (micStatus) micStatus.textContent = "Listening...";
        if (audioStatus) audioStatus.textContent = "Recording";

        try {
            const constraints = { audio: { deviceId: selectedMicId ? { exact: selectedMicId } : undefined } };
            micStream = await navigator.mediaDevices.getUserMedia(constraints);
            console.log("✅ Microphone access granted.");

            // Initialize VAD processor if available (requires separate VAD library)
            // This is a simplified simulation using ScriptProcessorNode and basic threshold
            // For a proper VAD, integrate a library like Silero VAD JS
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(micStream);
            const processor = audioContext.createScriptProcessor(1024, 1, 1); // Buffer size, input channels, output channels

            let isActive = false;
            let silenceStart = 0;
            const silenceThreshold = 2000; // 2 seconds of silence to trigger end

            processor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                let sum = 0;
                for (let i = 0; i < inputData.length; i++) {
                    const sample = Math.abs(inputData[i] * volumeLevel); // Apply volume multiplier
                    sum += sample * sample;
                }
                const rms = Math.sqrt(sum / inputData.length);
                const isSpeechDetected = rms > vadThreshold;

                // Update audio status based on VAD
                if (isSpeechDetected && !isActive) {
                    console.log("🎤 Speech started detected by client-side VAD simulation.");
                    isActive = true;
                    if (audioStatus) audioStatus.textContent = "Listening (VAD)";
                    // Send start signal to backend if needed (usually happens on first chunk)
                    showVoiceCircle();
                    // Reset silence timer
                    silenceStart = Date.now();
                } else if (!isSpeechDetected && isActive) {
                    // Check if silence duration exceeds threshold
                    if (Date.now() - silenceStart > silenceThreshold) {
                        console.log("🤫 Speech ended detected by client-side VAD simulation after timeout.");
                        isActive = false;
                        if (audioStatus) audioStatus.textContent = "Processing...";
                        // Stop recording automatically
                        toggleMic();
                    }
                } else if (isSpeechDetected && isActive) {
                    // Reset silence timer if speech is detected again
                    silenceStart = Date.now();
                }

                // Send audio data to WebSocket (if connected)
                if (ws && ws.readyState === WebSocket.OPEN) {
                    // Convert float32 array to Int16 for transmission (or keep as float32 if backend handles it)
                    const int16Array = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        int16Array[i] = Math.max(-1, Math.min(1, inputData[i] * volumeLevel)) * 0x7FFF;
                    }
                    const audioChunk = String.fromCharCode.apply(null, new Uint8Array(int16Array.buffer));
                    try {
                        ws.send(JSON.stringify({
                            type: 'audio_chunk',
                            audio_data: btoa(audioChunk), // Encode as base64
                            session_id: SESSION_ID
                        }));
                    } catch (sendError) {
                        console.error("Error sending audio chunk:", sendError);
                    }
                }
            };

            source.connect(processor);
            processor.connect(audioContext.destination);
            vadProcessor = processor; // Store reference to stop later

        } catch (err) {
            console.error("❌ Microphone access denied or failed:", err);
            showNotification("Microphone access denied. Please allow access in your browser settings.", "error");
            isRecording = false;
            if (micBtn) {
                micBtn.classList.remove('pulse', 'bg-red-600');
                micBtn.classList.add('bg-indigo-600');
            }
            if (micStatus) micStatus.textContent = "Mic denied";
            if (audioStatus) audioStatus.textContent = "Error";
        }
    }
}

// --- UI Utilities ---
function showVoiceCircle() {
    const circle = document.getElementById('voice-circle');
    if (circle) {
        circle.classList.add('active');
        circle.style.display = 'block';
    }
}

function hideVoiceCircle() {
    const circle = document.getElementById('voice-circle');
    if (circle) {
        circle.classList.remove('active');
        // Optional: Hide after animation completes to avoid flickering if show/hide happen quickly
        // setTimeout(() => { if (!circle.classList.contains('active')) circle.style.display = 'none'; }, 2000);
    }
}

function showNotification(message, type = 'info') {
    // Create or reuse a notification element
    let notification = document.getElementById('globalNotification');
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'globalNotification';
        notification.className = 'fixed top-4 right-4 p-4 rounded-lg shadow-lg text-white z-50 max-w-xs';
        document.body.appendChild(notification);
    }

    // Set content and style based on type
    notification.textContent = message;
    notification.className = 'fixed top-4 right-4 p-4 rounded-lg shadow-lg text-white z-50 max-w-xs ';
    switch (type) {
        case 'success':
            notification.classList.add('bg-green-600');
            break;
        case 'error':
            notification.classList.add('bg-red-600');
            break;
        case 'warning':
            notification.classList.add('bg-yellow-600', 'text-gray-900');
            break;
        default: // 'info'
            notification.classList.add('bg-blue-600');
    }

    // Show notification
    notification.style.display = 'block';

    // Auto-hide after 5 seconds
    setTimeout(() => {
        notification.style.display = 'none';
    }, 5000);
}

// --- Settings Modal ---
async function initializeSettingsModal() {
    console.log("🔧 Initializing settings modal...");
    const micSelect = document.getElementById('micSelect');
    const outputSelect = document.getElementById('outputSelect');
    const saveBtn = document.getElementById('saveAudioSettingsBtn');
    const testMicBtn = document.getElementById('testMicBtn');
    const testAudioBtn = document.getElementById('testAudioBtn');
    const vadEnabled = document.getElementById('vadEnabled');
    const vadThreshold = document.getElementById('vadThreshold');
    const vadThresholdValue = document.getElementById('vadThresholdValue');
    const volumeLevelSlider = document.getElementById('volumeLevel');
    const volumeLevelValue = document.getElementById('volumeLevelValue');
    const speakerVolumeSlider = document.getElementById('speakerVolume');
    const speakerVolumeValue = document.getElementById('speakerVolumeValue');

    if (!micSelect || !outputSelect || !saveBtn) {
        console.error("❌ Settings modal elements not found.");
        return;
    }

    // Load saved settings from localStorage or use defaults
    const savedSettings = JSON.parse(localStorage.getItem('audioSettings') || '{}');
    if (savedSettings.micId) selectedMicId = savedSettings.micId;
    if (savedSettings.outputId) selectedOutputId = savedSettings.outputId;
    if (savedSettings.vadEnabled !== undefined) isVADActive = savedSettings.vadEnabled;
    if (savedSettings.vadThreshold !== undefined) vadThreshold.value = savedSettings.vadThreshold;
    if (savedSettings.volumeLevel !== undefined) volumeLevelSlider.value = savedSettings.volumeLevel;
    if (savedSettings.speakerVolume !== undefined) speakerVolumeSlider.value = savedSettings.speakerVolume;

    // Update UI elements based on loaded settings
    vadEnabled.checked = isVADActive;
    vadThresholdValue.textContent = vadThreshold.value;
    volumeLevelValue.textContent = volumeLevelSlider.value;
    speakerVolumeValue.textContent = speakerVolumeSlider.value;

    // Populate microphone and output device lists
    try {
        const devices = await navigator.mediaDevices.enumerateDevices();
        const mics = devices.filter(device => device.kind === 'audioinput');
        const outputs = devices.filter(device => device.kind === 'audiooutput'); // Not directly used for output in this basic setup, but list for selection

        micSelect.innerHTML = '<option value="">Default Microphone</option>';
        mics.forEach(mic => {
            const option = document.createElement('option');
            option.value = mic.deviceId;
            option.textContent = mic.label || `Microphone ${micSelect.length}`;
            if (mic.deviceId === selectedMicId) option.selected = true;
            micSelect.appendChild(option);
        });

        outputSelect.innerHTML = '<option value="">Default Speaker</option>';
        outputs.forEach(output => {
            const option = document.createElement('option');
            option.value = output.deviceId;
            option.textContent = output.label || `Speaker ${outputSelect.length}`;
            if (output.deviceId === selectedOutputId) option.selected = true;
            outputSelect.appendChild(option);
        });

    } catch (err) {
        console.error("❌ Error getting media devices:", err);
        showNotification("Could not access audio devices. Please check permissions.", "error");
    }

    // Event Listeners for Sliders
    if (vadThreshold) {
        vadThreshold.addEventListener('input', () => {
            vadThresholdValue.textContent = vadThreshold.value;
        });
    }
    if (volumeLevelSlider) {
        volumeLevelSlider.addEventListener('input', () => {
            volumeLevelValue.textContent = volumeLevelSlider.value;
        });
    }
    if (speakerVolumeSlider) {
        speakerVolumeSlider.addEventListener('input', () => {
            speakerVolumeValue.textContent = speakerVolumeSlider.value;
        });
    }

    // Save Settings
    saveBtn.addEventListener('click', () => {
        selectedMicId = micSelect.value || null;
        selectedOutputId = outputSelect.value || null;
        isVADActive = vadEnabled.checked;
        vadThresholdValue = parseFloat(vadThreshold.value);
        volumeLevel = parseFloat(volumeLevelSlider.value);
        speakerVolume = parseFloat(speakerVolumeSlider.value); // Store for potential future use

        const settingsToSave = {
            micId: selectedMicId,
            outputId: selectedOutputId,
            vadEnabled: isVADActive,
            vadThreshold: vadThresholdValue,
            volumeLevel: volumeLevel,
            speakerVolume: speakerVolume
        };
        localStorage.setItem('audioSettings', JSON.stringify(settingsToSave));
        configSaved = true;
        showNotification("Audio settings saved!", "success");
        document.getElementById('settingsModal').classList.add('hidden');
    });

    // Test Buttons (Basic Implementation)
    testMicBtn.addEventListener('click', async () => {
        console.log("🔍 Testing microphone...");
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: { deviceId: selectedMicId ? { exact: selectedMicId } : undefined } });
            showNotification("Microphone test successful!", "success");
            stream.getTracks().forEach(track => track.stop()); // Stop immediately after test
        } catch (err) {
            console.error("❌ Microphone test failed:", err);
            showNotification("Microphone test failed. Check device permissions.", "error");
        }
    });

    testAudioBtn.addEventListener('click', () => {
        console.log("🔍 Testing audio output...");
        // Basic beep using Web Audio API
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        oscillator.type = 'sine';
        oscillator.frequency.value = 440; // A4 note
        gainNode.gain.value = 0.1; // Low volume
        oscillator.start();
        setTimeout(() => {
            oscillator.stop();
            showNotification("Audio output test complete.", "info");
        }, 500);
    });
}

// --- User Info & History Functions ---
async function loadUserInfo() {
    try {
        console.log("👤 Loading user info...");
        const response = await fetch('/api/user/profile');
        if (response.ok) {
            const userData = await response.json();
            const userEmailElement = document.getElementById('currentUserEmail');
            if (userEmailElement) {
                userEmailElement.textContent = userData.email;
                console.log("✅ User info loaded:", userData.email);
            }
        } else {
            console.error('❌ Failed to load user profile:', response.status);
            const userEmailElement = document.getElementById('currentUserEmail');
            if (userEmailElement) {
                userEmailElement.textContent = 'Not logged in';
            }
        }
    } catch (error) {
        console.error('❌ Failed to load user info:', error);
        const userEmailElement = document.getElementById('currentUserEmail');
        if (userEmailElement) {
            userEmailElement.textContent = 'Error loading';
        }
    }
}

async function loadConversationHistory() {
    try {
        console.log("📚 Loading conversation history...");
        // Fetch conversations for the *current* user only via the backend endpoint
        const response = await fetch('/api/user/conversations');

        if (response.ok) {
            const conversations = await response.json();
            console.log("✅ Loaded conversations:", conversations.length);
            displayHistoryList(conversations); // Pass the data to the display function
        } else {
            console.error('❌ Failed to load conversation history:', response.status);
            displayHistoryList([]); // Show empty state if failed
        }
    } catch (error) {
        console.error('❌ Failed to load conversation history:', error);
        displayHistoryList([]); // Show empty state if error
    }
}

function displayHistoryList(conversations) {
    const historyList = document.getElementById('historyList');
    if (!historyList) {
        console.error("❌ historyList element not found");
        return;
    }

    if (!conversations || conversations.length === 0) {
        // Show empty state if no conversations
        historyList.innerHTML = `<div class="text-gray-400 text-center py-8">
                                   <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 mx-auto mb-2 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                     <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                                   </svg>
                                   <p>No conversations yet</p>
                                   <p class="text-xs mt-2">Start chatting to see history here</p>
                                 </div>`;
        return;
    }

    // Sort conversations by timestamp (newest first) - redundant if backend already sorts, but safe
    conversations.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    // Generate HTML for each conversation item and insert into the list
    // *** REMOVED .slice(0, 20) TO DISPLAY ALL CONVERSATIONS ***
    historyList.innerHTML = conversations.map(conv => {
        // Escape HTML for safety
        const escapeHtml = (unsafe) => {
            return unsafe
                .replace(/&/g, "&amp;")
                .replace(/</g, "<")
                .replace(/>/g, ">")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        };

        return `<div class="history-item p-3 rounded-lg cursor-pointer bg-gray-800 hover:bg-gray-700 transition-colors" data-conv-id="${conv.id}">
                    <div class="flex justify-between items-start mb-2">
                        <span class="text-xs text-gray-400">${new Date(conv.timestamp).toLocaleDateString()}</span>
                        <span class="text-xs text-indigo-400">${new Date(conv.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div class="text-sm font-medium truncate mb-1" title="${escapeHtml(conv.user_message)}">
                        ${escapeHtml(conv.user_message.substring(0, 50))}${conv.user_message.length > 50 ? '...' : ''}
                    </div>
                    <div class="text-xs text-gray-400 truncate" title="${escapeHtml(conv.ai_message)}">
                        ${escapeHtml(conv.ai_message.substring(0, 60))}${conv.ai_message.length > 60 ? '...' : ''}
                    </div>
                 </div>`;
    }).join('');

    // Add click handlers to each history item to load that specific conversation
    historyList.querySelectorAll('.history-item').forEach(item => {
        item.addEventListener('click', () => {
            loadConversation(item.dataset.convId);
        });
    });
    console.log("✅ History list updated with", conversations.length, "conversations");
}

async function loadConversation(conversationId) {
    try {
        console.log("📖 Loading conversation:", conversationId);
        const response = await fetch(`/api/conversations/${conversationId}`);
        if (response.ok) {
            const conversation = await response.json();
            displayConversation(conversation);
            showNotification('Conversation loaded', 'info');
        } else {
            console.error('❌ Failed to load conversation:', response.status);
            showNotification('Failed to load conversation', 'error');
        }
    } catch (error) {
        console.error('❌ Failed to load conversation:', error);
        showNotification('Error loading conversation', 'error');
    }
}

function displayConversation(conversation) {
    console.log("🖼️ Displaying conversation:", conversation);
    const pane = document.getElementById('conversationHistory');
    if (!pane) {
        console.error("❌ Conversation pane not found for display.");
        return;
    }

    // Clear current conversation display
    pane.innerHTML = '';

    // Add user message
    if (conversation.user_message) {
        addMessageToConversation('user', conversation.user_message, new Date(conversation.timestamp).toLocaleTimeString());
    }
    // Add AI message
    if (conversation.ai_message) {
        addMessageToConversation('ai', conversation.ai_message, new Date(conversation.timestamp).toLocaleTimeString());
    }

    // Scroll to bottom
    pane.scrollTop = pane.scrollHeight;
}

// --- System Status ---
async function checkSystemStatus() {
    try {
        console.log("🔧 Checking system status...");
        const response = await fetch('/api/status');
        if (response.ok) {
            const status = await response.json();
            updateModelStatus(status);
        } else {
            console.error('❌ Failed to check system status:', response.status);
            updateModelStatus({ models_loaded: false });
        }
    } catch (error) {
        console.error('❌ Failed to check system status:', error);
        updateModelStatus({ models_loaded: false });
    }
}

function updateModelStatus(status) {
    const element = document.getElementById('modelStatus');
    if (!element) {
        console.error("❌ modelStatus element not found");
        return;
    }

    if (status.models_loaded) {
        const loadedModels = [];
        if (status.whisper_loaded) loadedModels.push('Speech');
        if (status.llm_loaded) loadedModels.push('LLM');
        if (status.rag_loaded) loadedModels.push('RAG');
        if (loadedModels.length > 0) {
            element.textContent = loadedModels.join(', ');
            element.className = 'text-green-400';
            console.log("✅ Models status updated:", loadedModels.join(', '));
        } else {
            element.textContent = 'Models Loading...';
            element.className = 'text-yellow-400';
            console.log("⏳ Models still loading...");
        }
    } else {
        element.textContent = 'Checking...';
        element.className = 'text-yellow-400';
        console.log("⏳ Models status: Checking...");
    }
}

// --- Duration Timer ---
function startDurationTimer() {
    sessionStartTime = new Date();
    console.log("⏱️ Session started at:", sessionStartTime);

    setInterval(() => {
        if (sessionStartTime) {
            const now = new Date();
            const diffMs = now - sessionStartTime;
            const diffSecs = Math.floor(diffMs / 1000);
            const hours = Math.floor(diffSecs / 3600);
            const minutes = Math.floor((diffSecs % 3600) / 60);
            const seconds = diffSecs % 60;

            const formattedTime = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
            const durationElement = document.getElementById('sessionDuration');
            if (durationElement) {
                durationElement.textContent = formattedTime;
            }
        }
    }, 1000); // Update every second
}

// --- Audio Visualization Chart (using Chart.js) ---
function initializeAudioChart() {
    const micCanvas = document.getElementById('micCanvas');
    const outputCanvas = document.getElementById('outputCanvas');

    if (micCanvas) {
        // This is a placeholder for actual audio level visualization
        // You would connect this to the audio processing nodes
        const micCtx = micCanvas.getContext('2d');
        micCtx.fillStyle = '#4F46E5'; // Indigo
        micCtx.fillRect(0, micCanvas.height / 2 - 5, micCanvas.width, 10); // Simple bar
    }
    if (outputCanvas) {
        const outCtx = outputCanvas.getContext('2d');
        outCtx.fillStyle = '#10B981'; // Emerald
        outCtx.fillRect(0, outputCanvas.height / 2 - 5, outputCanvas.width, 10); // Simple bar
    }
    // Note: Implementing real-time audio level visualization requires more complex integration
    // with the AudioContext and AnalyserNode, which is beyond the scope of this basic example.
}

// --- Debugging Functions ---
function debugConversationHistory() {
    console.log("=== CONVERSATION HISTORY DEBUG ===");
    const pane = document.getElementById('conversationHistory');
    if (!pane) {
        console.error("No conversation pane found");
        return;
    }
    console.log("Number of messages in pane:", pane.children.length);
    Array.from(pane.children).forEach((msg, index) => {
        console.log(`Message ${index}:`, msg.textContent.substring(0, 100) + "...");
    });
}

function debugWebSocket() {
    console.log("=== WEBSOCKET DEBUG ===");
    console.log("WebSocket state:", ws ? ws.readyState : "no websocket");
    console.log("Active generation ID:", activeGenId);
    console.log("Audio playing:", isAudioCurrentlyPlaying);
    console.log("Queue length:", audioPlaybackQueue.length);
    console.log("Interrupt flags:", { interruptRequested, interruptInProgress });
    if (ws && ws.readyState === WebSocket.OPEN) {
        const testMsg = {
            type: 'test',
            message: 'Debug test',
            session_id: SESSION_ID
        };
        console.log("Sending test message:", testMsg);
        ws.send(JSON.stringify(testMsg));
    }
}

function testMessageDisplay() {
    console.log("🧪 Testing message display...");
    // Test adding a user message
    addMessageToConversation('user', 'Test user message from console');
    // Test adding an AI message
    addMessageToConversation('ai', 'Test AI message from console');
    // Check if they appeared
    debugConversationHistory();
}

function testFullFlow() {
    console.log("=== TESTING FULL FLOW ===");
    // Test 1: Check conversation pane
    debugConversationHistory();
    // Test 2: Test WebSocket connection
    if (ws && ws.readyState === WebSocket.OPEN) {
        console.log("✅ WebSocket is connected");
        // Send a test message
        const testMsg = {
            type: 'test',
            message: 'Full flow test',
            session_id: SESSION_ID,
            timestamp: new Date().toISOString()
        };
        console.log("Sending test message:", testMsg);
        ws.send(JSON.stringify(testMsg));
        // Send a real message after 2 seconds
        setTimeout(() => {
            console.log("Sending real text message...");
            sendTextMessage("Test message from debug function");
        }, 2000);
    } else {
        console.error("❌ WebSocket not connected");
    }
}

function checkConversationStyles() {
    console.log("=== CHECKING CONVERSATION STYLES ===");
    const pane = document.getElementById('conversationHistory');
    if (!pane) {
        console.error("No conversation pane found");
        return;
    }
    const computedStyle = window.getComputedStyle(pane);
    console.log("Pane styles:");
    console.log("- display:", computedStyle.display);
    console.log("- visibility:", computedStyle.visibility);
    console.log("- opacity:", computedStyle.opacity);
    console.log("- height:", computedStyle.height);
    console.log("- overflow:", computedStyle.overflow);
    console.log("- position:", computedStyle.position);

    // Check if parent elements are visible
    let parent = pane.parentElement;
    let level = 0;
    while (parent && level < 5) {
        const parentStyle = window.getComputedStyle(parent);
        console.log(`Parent ${level} (${parent.tagName}.${parent.className}):`);
        console.log(` - display: ${parentStyle.display}`);
        console.log(` - visibility: ${parentStyle.visibility}`);
        console.log(` - opacity: ${parentStyle.opacity}`);
        parent = parent.parentElement;
        level++;
    }
}

// --- Sending Messages ---
function sendTextMessage(text) {
    if (!text || text.trim() === '') {
        console.error("❌ Empty message text provided");
        return;
    }

    // Add user message to UI immediately
    addMessageToConversation('user', text);

    // Send to WebSocket server
    if (ws && ws.readyState === WebSocket.OPEN) {
        console.log("📤 Sending text message via WebSocket:", text.substring(0, 50) + (text.length > 50 ? '...' : ''));
        ws.send(JSON.stringify({
            type: 'text',
            text: text,
            session_id: SESSION_ID
        }));
    } else {
        console.error("❌ WebSocket not open, cannot send message.");
        showNotification("Not connected to server. Cannot send message.", "error");
    }
}

// --- Global Functions for Console Debugging ---
window.loadConversationHistory = loadConversationHistory; // Make available globally if needed elsewhere
window.displayHistoryList = displayHistoryList;
window.requestInterrupt = requestInterrupt;
window.clearAudioPlayback = clearAudioPlayback;