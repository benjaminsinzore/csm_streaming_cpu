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

let audioPlaybackQueue = [];
let audioDataHistory = [];
let micAnalyser, micContext;
let activeGenId = 0;

// ==================== DEBUG FUNCTIONS ====================

function debugConversationHistory() {
  console.log("=== DEBUG CONVERSATION HISTORY ===");
  const pane = document.getElementById('conversationHistory');
  console.log("Conversation pane element:", pane);
  console.log("Pane exists:", !!pane);
  if (pane) {
    console.log("Pane children:", pane.children.length);
    console.log("Pane innerHTML:", pane.innerHTML);
    console.log("Pane classList:", pane.classList);
  } else {
    console.error("❌ Conversation history pane not found!");
  }
}

function debugWebSocket() {
  console.log("=== WEBSOCKET DEBUG ===");
  console.log("WebSocket state:", ws ? ws.readyState : "no websocket");
  console.log("Active generation ID:", activeGenId);
  console.log("Audio playing:", isAudioCurrentlyPlaying);
  console.log("Queue length:", audioPlaybackQueue.length);
  console.log("Interrupt flags:", {interruptRequested, interruptInProgress});
  
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

// Make all debug functions available globally
window.debugConversationHistory = debugConversationHistory;
window.debugWebSocket = debugWebSocket;
window.testMessageDisplay = testMessageDisplay;
window.testFullFlow = testFullFlow;
window.addMessageToConversation = addMessageToConversation;
window.sendTextMessage = sendTextMessage;

// ==================== MAIN FUNCTIONS ====================

function createPermanentVoiceCircle() {
  if (document.getElementById('voice-circle')) return;
  const style = document.createElement('style');
  style.textContent = `
    #voice-circle{
      position:fixed;top:50%;left:50%;
      width:180px;height:180px;border-radius:50%;
      background:rgba(99,102,241,.20);
      transform:translate(-50%,-50%) scale(var(--dynamic-scale,1));
      pointer-events:none;z-index:50;
      transition:background-color .35s ease;
    }
    #voice-circle.active{
      animation:pulse-circle 2s infinite alternate ease-in-out;
    }
    @keyframes pulse-circle{
      0%{background:rgba(99,102,241,.55)}
      100%{background:rgba(99,102,241,.20)}
    }`;
  document.head.appendChild(style);

  const c = document.createElement('div');
  c.id='voice-circle';
  document.body.appendChild(c);
  console.log("Created permanent voice circle");
}

function showVoiceCircle() {
  const c=document.getElementById('voice-circle')||createPermanentVoiceCircle();
  c.classList.add('active');
}

function hideVoiceCircle() {
  const c=document.getElementById('voice-circle');
  if (c){ c.classList.remove('active'); c.style.setProperty('--dynamic-scale',1); }
}

function showNotification(msg, type='info'){
  const n=document.createElement('div');
  n.className=`fixed bottom-4 right-4 px-4 py-3 rounded-lg shadow-lg z-50
               ${type==='success'?'bg-green-600':
                 type==='error'  ?'bg-red-600':'bg-indigo-600'}`;
  n.textContent=msg;
  document.body.appendChild(n);
  setTimeout(()=>{n.classList.add('opacity-0');
                  setTimeout(()=>n.remove(),500)},3000);
}

function addMessageToConversation(sender, text) {
  console.log(`🔄 Attempting to add ${sender} message:`, text);
  
  // Validate input
  if (!text || text.trim() === '') {
    console.error("❌ Empty message text provided");
    return;
  }
  
  const pane = document.getElementById('conversationHistory');
  if (!pane) {
    console.error("❌ Conversation history pane not found! Check if element exists in HTML.");
    
    // Try to find the element with different selectors
    const alternativeSelectors = [
      '#conversationHistory',
      '.conversation-container div',
      '[id*="conversation"]',
      '[class*="conversation"]'
    ];
    
    for (const selector of alternativeSelectors) {
      const element = document.querySelector(selector);
      if (element) {
        console.log(`Found alternative element with selector: ${selector}`, element);
      }
    }
    return;
  }

  console.log("✅ Found conversation pane, creating message element...");
  
  try {
    const box = document.createElement('div');
    box.className = `p-3 mb-3 rounded-lg text-sm ${
      sender === 'user' ? 'bg-gray-800 ml-2' : 'bg-indigo-900 mr-2'
    }`;
    
    const timestamp = new Date().toLocaleTimeString();
    const escapedText = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;')
      .replace(/\n/g, '<br>');

    box.innerHTML = `
      <div class="flex items-start mb-2">
        <div class="w-6 h-6 rounded-full flex items-center justify-center
             ${sender === 'user' ? 'bg-gray-300 text-gray-800' : 'bg-indigo-500 text-white'}">
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
    
    console.log(`✅ Added ${sender} message to conversation. Total messages: ${pane.children.length}`);
    
    // Verify the message was actually added
    setTimeout(() => {
      const lastChild = pane.lastElementChild;
      if (lastChild && lastChild.textContent.includes(text.substring(0, 20))) {
        console.log("✅ Message verified in DOM");
      } else {
        console.error("❌ Message not found in DOM after addition");
      }
    }, 100);
    
  } catch (error) {
    console.error("❌ Error adding message to conversation:", error);
  }
}

function connectWebSocket() {
  if (reconnecting && reconnectAttempts >= maxReconnectAttempts) {
    console.error("Maximum reconnect attempts reached. Please refresh the page.");
    showNotification("Connection lost. Please refresh the page.", "error");
    return;
  }

  if (ws && ws.readyState !== WebSocket.CLOSED && ws.readyState !== WebSocket.CLOSING) {
    try {
      ws.close();
    } catch (e) {
      console.warn("Error closing existing WebSocket:", e);
    }
  }

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws?session_id=${SESSION_ID}`);
  window.ws = ws;

  const connLbl = document.getElementById('connectionStatus');
  if (connLbl) {
    connLbl.textContent = reconnecting ? 'Reconnecting…' : 'Connecting…';
    connLbl.className = 'text-yellow-500';
  }

  ws.onopen = () => {
    console.log("✅ WebSocket connected successfully");
    if (connLbl) {
      connLbl.textContent = 'Connected';
      connLbl.className = 'text-green-500';
    }
    
    reconnecting = false;
    reconnectAttempts = 0;
    
    if (!reconnecting) {
      addMessageToConversation('ai', 'WebSocket connected. Ready for voice or text.');
    } else {
      showNotification("Reconnected successfully", "success");
    }
  };

  ws.onclose = (event) => {
    console.log("❌ WebSocket closed with code:", event.code, "reason:", event.reason);
    if (connLbl) {
      connLbl.textContent = 'Disconnected';
      connLbl.className = 'text-red-500';
    }

    clearAudioPlayback();
    
    if (event.code !== 1000 && event.code !== 1001) {
      reconnecting = true;
      reconnectAttempts++;
      
      const delay = Math.min(1000 * Math.pow(1.5, reconnectAttempts), 1000);
      console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);
      
      setTimeout(connectWebSocket, delay);
    }
  };

  ws.onerror = (error) => {
    console.error("❌ WebSocket error:", error);
    if (connLbl) {
      connLbl.textContent = 'Error';
      connLbl.className = 'text-red-500';
    }
  };

  ws.onmessage = (e) => {
    console.log("📨 RAW WebSocket message received:", e.data);
    try {
      const data = JSON.parse(e.data);
      console.log("📨 Parsed WebSocket message type:", data.type, "data:", data);
      handleWebSocketMessage(data);
    } catch (err) {
      console.error("❌ Error parsing WebSocket message:", err, "Raw data:", e.data);
    }
  };
}




function sendTextMessage(txt) {
  if (!txt.trim()) {
    console.log("Empty message, ignoring");
    return;
  }
  
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    showNotification("Not connected", "error");
    return;
  }

  console.log("=== SENDING TEXT MESSAGE ===");
  console.log("Message:", txt);
  console.log("WebSocket state:", ws ? ws.readyState : "no websocket");

  // Create the message payload
  const messageData = {
    type: 'text_message',
    text: txt,
    session_id: SESSION_ID,
    timestamp: new Date().toISOString()
  };
  
  console.log("Sending message data:", messageData);
  
  try {
    // Add user message to UI immediately - FORCE DISPLAY
    console.log("🔄 Adding user message to conversation...");
    addMessageToConversation('user', txt);
    
    // Double-check that the message was added
    setTimeout(() => {
      const pane = document.getElementById('conversationHistory');
      if (pane) {
        console.log("✅ User message should be visible. Total messages in pane:", pane.children.length);
        // Force scroll to make sure it's visible
        pane.scrollTop = pane.scrollHeight;
      }
    }, 100);
    
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
    
    // Send message to server
    ws.send(JSON.stringify(messageData));
    
    console.log("✅ Text message sent successfully");
    
  } catch (error) {
    console.error("❌ Error sending message:", error);
    showNotification("Error sending message", "error");
  }
}




function clearAudioPlayback() {
  console.log("FORCEFULLY CLEARING AUDIO PLAYBACK");
  
  interruptRequested = true;
  interruptInProgress = true;
  
  try {
    console.log(`Clearing queue with ${audioPlaybackQueue.length} items`);
    audioPlaybackQueue = [];
    
    activeGenId = 0;
    
    if (currentAudioSource) {
      console.log("Stopping active audio source");
      
      try {
        if (currentAudioSource.disconnect) {
          currentAudioSource.disconnect();
        }
      } catch (e) {
        console.warn("Error disconnecting audio source:", e);
      }
      
      try {
        if (currentAudioSource.stop) {
          currentAudioSource.stop(0);
        }
      } catch (e) {
        console.warn("Error stopping audio source:", e);
      }
      
      currentAudioSource = null;
    }
    
    try {
      if (audioContext) {
        const oldContext = audioContext;
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        window.audioContext = audioContext;
        
        try {
          oldContext.close();
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
  } catch (err) {
    console.error("Error clearing audio:", err);
  }
  
  isAudioCurrentlyPlaying = false;
  hideVoiceCircle();
  
  console.log("Audio playback cleared successfully");
  
  setTimeout(() => {
    interruptInProgress = false;
  }, 300);
}

function requestInterrupt() {
  console.log("User requested interruption");
  
  if (interruptInProgress) {
    console.log("Interrupt already in progress - force clearing again");
    clearAudioPlayback();
    return false;
  }
  
  interruptRequested = true;
  interruptInProgress = true;
  
  showNotification("Interrupting...", "info");
  
  clearAudioPlayback();
  
  const interruptBtn = document.getElementById('interruptBtn');
  if (interruptBtn) {
    interruptBtn.classList.add('bg-red-800');
    setTimeout(() => {
      interruptBtn.classList.remove('bg-red-800');
    }, 300);
  }
  
  if (ws && ws.readyState === WebSocket.OPEN) {
    console.log("Sending interrupt request to server");
    try {
      ws.send(JSON.stringify({
        type: 'interrupt',
        immediate: true
      }));
    } catch (error) {
      console.error("Error sending interrupt request:", error);
    }
    
    setTimeout(() => {
      if (interruptInProgress) {
        console.log("No interrupt confirmation received from server, resetting state");
        interruptInProgress = false;
      }
    }, 2000);
    
    return true;
  } else {
    console.warn("WebSocket not available for interrupt request");
    setTimeout(() => {
      interruptInProgress = false;
    }, 500);
    return false;
  }
}

function handleWebSocketMessage(d) {
  console.log("📨 Received WebSocket message:", d.type, d);
  
  switch(d.type) {
    case 'transcription':
      console.log("🎤 Processing transcription...");
      addMessageToConversation('user', d.text);
      showVoiceCircle();
      break;
      
    case 'response':
      console.log("🤖 Processing AI response...");
      addMessageToConversation('ai', d.text);
      showVoiceCircle();
      break;
      
    case 'audio_chunk':
      console.log("🔊 Audio chunk received - genId:", d.gen_id, "activeGenId:", activeGenId);
      
      if (activeGenId === 0 && d.gen_id) {
        activeGenId = d.gen_id;
        console.log("🎯 Setting active generation ID to:", activeGenId);
      }
      
      if (activeGenId === 0 || d.gen_id === activeGenId) {
        queueAudioForPlayback(d.audio, d.sample_rate, d.gen_id || 0);
        showVoiceCircle();
      } else {
        console.log("🚫 Ignoring audio chunk - generation ID mismatch");
      }
      break;
      
    case 'audio_status':
      console.log("🔊 Audio status:", d.status, "genId:", d.gen_id);
      
      if (d.status === 'generating') {
        console.log("🔄 New audio generation starting");
        interruptRequested = false;
        interruptInProgress = false;
        
        if (d.gen_id) {
          activeGenId = d.gen_id;
          console.log("🎯 Active generation set to:", activeGenId);
        }
        
        showVoiceCircle();
      } 
      else if (d.status === 'first_chunk') {
        console.log("🎵 First audio chunk ready");
        showVoiceCircle();
      }
      else if (d.status === 'complete') {
        console.log("✅ Audio generation complete");
        activeGenId = 0;
        if (!isAudioCurrentlyPlaying) {
          hideVoiceCircle();
        }
      } 
      else if (d.status === 'interrupted' || d.status === 'interrupt_acknowledged') {
        console.log("⏹️ Audio interrupted by server");
        clearAudioPlayback();
      }
      break;
      
    case 'status':
      console.log("ℹ️ Status:", d.message);
      if (d.message === 'Thinking...') {
        showVoiceCircle();
      }
      break;
      
    case 'error':
      console.error("❌ Error:", d.message);
      showNotification(d.message, 'error');
      hideVoiceCircle();
      break;
      
    case 'vad_status':
      console.log("🎤 VAD Status:", d.status);
      if (d.status === 'speech_started') {
        showVoiceCircle();
      }
      break;

    case 'test_response':
      console.log("✅ Test response received:", d.message);
      showNotification(d.message, 'success');
      break;
      
    default:
      console.log("❓ Unknown message type:", d.type);
  }
}

function queueAudioForPlayback(arr, sr, genId = 0) {
  console.log("🎵 Queueing audio chunk - genId:", genId, "queue length:", audioPlaybackQueue.length);
  
  if (interruptRequested || interruptInProgress) {
    console.log("🚫 Interrupt active - skipping audio chunk");
    return;
  }
  
  if (activeGenId === 0 && genId !== 0) {
    activeGenId = genId;
    console.log("🎯 First chunk - setting activeGenId to:", activeGenId);
  }
  
  if (activeGenId === 0 || genId === activeGenId) {
    audioPlaybackQueue.push({arr, sr, genId});
    
    if (!isAudioCurrentlyPlaying) {
      console.log("▶️ Starting audio playback from queue");
      processAudioPlaybackQueue();
    }
  } else {
    console.log("🚫 Generation mismatch - ignoring chunk");
  }
}

function processAudioPlaybackQueue() {
  if (!isAudioCurrentlyPlaying && audioPlaybackQueue.length > 0) {
    console.log("Starting first audio chunk - force clearing interrupt flags");
    interruptRequested = false;
    interruptInProgress = false;
  }
  
  if (interruptRequested || interruptInProgress) {
    console.log("Interrupt active - not processing audio queue");
    isAudioCurrentlyPlaying = false;
    hideVoiceCircle();
    return;
  }
  
  if (!audioPlaybackQueue.length) {
    console.log("📭 Audio queue empty, stopping playback");
    isAudioCurrentlyPlaying = false;
    hideVoiceCircle();
    currentAudioSource = null;
    return;
  }
  
  const interruptBtn = document.getElementById('interruptBtn');
  if (interruptBtn) {
    interruptBtn.disabled = false;
    interruptBtn.classList.remove('opacity-50');
  }
  
  console.log("Processing next audio chunk");
  isAudioCurrentlyPlaying = true;
  
  const {arr, sr, genId} = audioPlaybackQueue.shift();
  
  if (activeGenId !== 0 && genId !== activeGenId) {
    console.log(`Skipping stale chunk playback (gen ${genId} vs active ${activeGenId})`);
    processAudioPlaybackQueue();
    return;
  }
  
  playAudioChunk(arr, sr)
    .then(() => {
      if (!interruptRequested && !interruptInProgress) {
        processAudioPlaybackQueue();
      } else {
        console.log("interrupt active - stopping queue processing");
        isAudioCurrentlyPlaying = false;
        hideVoiceCircle();
      }
    })
    .catch(err => {
      console.error("Error in audio playback:", err);
      isAudioCurrentlyPlaying = false;
      hideVoiceCircle();
      
      setTimeout(() => {
        if (audioPlaybackQueue.length > 0 && !interruptRequested) {
          processAudioPlaybackQueue();
        }
      }, 200);
    });
}

async function playAudioChunk(audioArr, sampleRate) {
  if (interruptRequested || interruptInProgress) {
    console.log("🚫 Interrupt active - skipping playback");
    return Promise.resolve();
  }
  
  try {
    if (!audioContext) {
      console.log("🎵 Creating new audio context");
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      window.audioContext = audioContext;
    }
    
    if (audioContext.state === 'suspended') {
      console.log("🎵 Resuming suspended audio context");
      await audioContext.resume();
    }
    
    console.log("🎵 Playing audio chunk, length:", audioArr.length);
    
    const buf = audioContext.createBuffer(1, audioArr.length, sampleRate);
    buf.copyToChannel(new Float32Array(audioArr), 0);
    
    const src = audioContext.createBufferSource();
    src.buffer = buf;
    
    currentAudioSource = src;
    
    const an = audioContext.createAnalyser(); 
    an.fftSize = 256;
    src.connect(an); 
    an.connect(audioContext.destination); 
    
    console.log("🎵 Starting audio playback");
    src.start();
    
    const arr = new Uint8Array(an.frequencyBinCount);
    const circle = document.getElementById('voice-circle');
    
    function pump() {
      if (src !== currentAudioSource || interruptRequested || interruptInProgress) {
        return;
      }
      
      try {
        an.getByteFrequencyData(arr);
        const avg = arr.reduce((a,b) => a+b, 0) / arr.length;
        if (circle) {
          circle.style.setProperty('--dynamic-scale', (1+avg/255*1.5).toFixed(3));
        }
      } catch (e) {
        console.warn("Animation error:", e);
        return;
      }
      
      if (src.playbackState !== src.FINISHED_STATE) {
        requestAnimationFrame(pump);
      }
    }
    pump();
    
    return new Promise(resolve => {
      src.onended = () => {
        console.log("🎵 Audio chunk finished playing");
        if (src === currentAudioSource) {
          currentAudioSource = null;
        }
        resolve();
      };
    });
    
  } catch (error) {
    console.error("❌ Error playing audio chunk:", error);
    return Promise.resolve();
  }
}

async function startRecording() {
  if (isRecording) return;
  try {
    const constraints = {
      audio: selectedMicId ? {deviceId:{exact:selectedMicId}} : true
    };
    micStream = await navigator.mediaDevices.getUserMedia(constraints);

    if (!audioContext) audioContext = new (AudioContext||webkitAudioContext)();
    const src = audioContext.createMediaStreamSource(micStream);
    const proc = audioContext.createScriptProcessor(4096,1,1);
    src.connect(proc); proc.connect(audioContext.destination);

    proc.onaudioprocess = e => {
      const samples = Array.from(e.inputBuffer.getChannelData(0));
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.send(JSON.stringify({
            type:'audio',
            audio:samples,
            sample_rate:audioContext.sampleRate,
            session_id:SESSION_ID
          }));
        } catch (error) {
          console.error("Error sending audio data:", error);
          stopRecording();
        }
      }
    };

    window._micProcessor = proc;        
    isRecording = true;
    document.getElementById('micStatus').textContent = 'Listening…';
    showVoiceCircle();
  } catch (err) {
    console.error("Microphone access error:", err);
    showNotification('Microphone access denied','error');
  }
}

function stopRecording() {
  if (!isRecording) return;
  try {
    if (window._micProcessor) {
      window._micProcessor.disconnect();
      window._micProcessor = null;
    }
    if (micStream) {
      micStream.getTracks().forEach(t => t.stop());
      micStream = null;
    }
  } catch(e) {
    console.warn("Error stopping recording:", e);
  }
  isRecording = false;
  
  const micStatus = document.getElementById('micStatus');
  if (micStatus) {
    micStatus.textContent = 'Click to speak';
  }
  hideVoiceCircle();
}

async function setupChatUI() {
  document.documentElement.classList.add('bg-gray-950');
  document.documentElement.style.backgroundColor = '#030712';

  createPermanentVoiceCircle();
  connectWebSocket();

  initAudioLevelsChart();

  const txt = document.getElementById('textInput');
  const btn = document.getElementById('sendTextBtn');
  
  // Setup enhanced interrupt button
  const interruptBtn = document.createElement('button');
  interruptBtn.id = 'interruptBtn';
  interruptBtn.className = 'px-3 py-2 ml-2 bg-red-600 text-white rounded hover:bg-red-700 flex items-center transition duration-150';
  interruptBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clip-rule="evenodd" /></svg> Stop';
  interruptBtn.onclick = (e) => {
    e.preventDefault();
    try {
      requestInterrupt();
      interruptBtn.classList.add('bg-red-800', 'scale-95');
      setTimeout(() => interruptBtn.classList.remove('bg-red-800', 'scale-95'), 150);
    } catch (error) {
      console.error("Error in interrupt button handler:", error);
    }
  };
  interruptBtn.title = "Stop AI speech (Space or Esc)";
  interruptBtn.disabled = true;
  interruptBtn.classList.add('opacity-50', 'cursor-not-allowed');
  
  if (btn && btn.parentElement) {
    btn.parentElement.appendChild(interruptBtn);
  }
  
  // Add debug button for easier debugging of interrupt issues
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
  
  if (btn && btn.parentElement) {
    btn.parentElement.appendChild(debugBtn);
  }
  
  // Run the update function periodically
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
  
  if (btn) {
    btn.onclick = () => {
      try {
        sendTextMessage(txt.value);
      } catch (error) {
        console.error("Error in send button handler:", error);
      }
    };
  }
  
  if (txt) {
    txt.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        try {
          sendTextMessage(txt.value);
        } catch (error) {
          console.error("Error in text input handler:", error);
        }
      }
    });
  }
  
  const micBtn = document.getElementById('micToggleBtn');
  if (micBtn) {
    micBtn.addEventListener('click', () => {
      try {
        if (isRecording) stopRecording();
        else startRecording();
      } catch (error) {
        console.error("Error in mic button handler:", error);
      }
    });
  }
  
  // Add event listeners to detect keyboard interruptions
  document.addEventListener('keydown', e => {
    if ((e.code === 'Space' || e.code === 'Escape') && isAudioCurrentlyPlaying) {
      e.preventDefault();
      try {
        requestInterrupt();
        
        const interruptBtn = document.getElementById('interruptBtn');
        if (interruptBtn) {
          interruptBtn.classList.add('bg-red-800');
          setTimeout(() => {
            interruptBtn.classList.remove('bg-red-800');
          }, 200);
        }
      } catch (error) {
        console.error("Error in keyboard interrupt handler:", error);
      }
    }
  });
  
  // Initialize audio context
  if (!audioContext) {
    try {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      window.audioContext = audioContext;
    } catch (error) {
      console.error("Error creating audio context:", error);
      showNotification("Audio initialization failed. Please refresh the page.", "error");
    }
  }
  
  // Try to unlock audio context on user interaction
  ['click', 'touchstart', 'keydown'].forEach(ev =>
    document.addEventListener(ev, function unlock() {
      if (audioContext && audioContext.state === 'suspended') {
        try {
          audioContext.resume();
        } catch (error) {
          console.warn("Error resuming audio context:", error);
        }
      }
      document.removeEventListener(ev, unlock);
    })
  );

  console.log("Chat UI ready with enhanced interruption support");
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setupChatUI);
} else {
  setupChatUI();
}

function initAudioLevelsChart() {
  const ctx = document.getElementById('audioLevels');
  if (!ctx) return;
  
  try {
    if (audioLevelsChart) audioLevelsChart.destroy();
    
    const grad = ctx.getContext('2d').createLinearGradient(0, 0, 0, 100);
    grad.addColorStop(0, 'rgba(79,70,229,.6)');
    grad.addColorStop(1, 'rgba(79,70,229,.1)');
    
    audioLevelsChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array(30).fill(''),
        datasets: [{
          data: Array(30).fill(0),
          backgroundColor: grad,
          borderColor: 'rgba(99,102,241,1)',
          borderWidth: 2,
          tension: .4,
          fill: true,
          pointRadius: 0
        }]
      },
      options: {
        animation: false,
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: {display: false},
            grid: {color: 'rgba(255,255,255,.1)'}
          },
          x: {display: false, grid: {display: false}}
        },
        plugins: {
          legend: {display: false},
          tooltip: {enabled: false}
        },
        elements: {point: {radius: 0}}
      }
    });
  } catch (error) {
    console.error("Error initializing audio chart:", error);
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
    console.log(`  - display: ${parentStyle.display}`);
    console.log(`  - visibility: ${parentStyle.visibility}`);
    console.log(`  - opacity: ${parentStyle.opacity}`);
    parent = parent.parentElement;
    level++;
  }
}

window.checkConversationStyles = checkConversationStyles;