// Status color constants
const STATUS_COLORS = {
    connected: {
        status: '#10b981', // Green
        user: '#10b981'    // Green
    },
    loading: {
        status: '#f59e0b', // Orange
        user: '#f59e0b'    // Orange
    },
    disconnected: {
        status: '#ef4444', // Red
        user: '#ef4444'    // Red
    }
};

// Initialize conversations as empty array - will be loaded from server
let conversations = [];
let currentFilter = localStorage.getItem('conversationFilter') || 'all';
let isFetchingConversations = false;
let conversationsLastUpdated = null;

// Audio playback variables
let audioContext = null;
let audioChunks = [];
let currentAudioGenerationId = null;
let isPlayingAudio = false;
let currentAudioSource = null;

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

const MAX_PREVIEW_LENGTH = 80;

function getPreviewText(text) {
    if (!text) return '';
    const firstSentenceMatch = text.match(/^[^.!?]*[.!?](?=\s|$)|^[^.!?]+/);
    let firstSentence = firstSentenceMatch ? firstSentenceMatch[0] : text;
    if (firstSentence.length <= MAX_PREVIEW_LENGTH) {
        return firstSentence;
    } else {
        return firstSentence.substring(0, MAX_PREVIEW_LENGTH).trim() + '…';
    }
}

// Function to fetch conversations from server
async function fetchConversations() {
    if (isFetchingConversations) return conversations;
    
    isFetchingConversations = true;
    try {
        const sessionToken = getCookie('session_token');
        const headers = sessionToken ? {
            'Authorization': `Bearer ${sessionToken}`
        } : {};
        
        const response = await fetch('/api/user/conversations', {
            headers: headers
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // Transform server data to match our local format
            conversations = data.map(conv => ({
                id: conv.id,
                date: conv.timestamp,
                user_message: conv.user_message || '',
                ai_message: conv.ai_message || '',
                starred: false,
                audio_path: conv.audio_path || '',
                server_id: conv.id
            }));
            
            conversationsLastUpdated = new Date();
            console.log(`Loaded ${conversations.length} conversations from server`);
            
            return conversations;
        } else if (response.status === 401) {
            console.log('User not authenticated, using local conversations only');
            return conversations;
        } else {
            throw new Error(`HTTP ${response.status}`);
        }
    } catch (error) {
        console.error('Error fetching conversations:', error);
        return conversations;
    } finally {
        isFetchingConversations = false;
    }
}

// Function to refresh conversations
async function refreshConversations() {
    const spinner = document.createElement('div');
    spinner.innerHTML = `
        <div class="conversation-card" style="text-align: center; padding: 20px;">
            <div class="loading-spinner" style="margin: 0 auto 10px; width: 30px; height: 30px; border: 3px solid #f3f4f6; border-top: 3px solid #4f46e5; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <div style="color: #6b7280;">Loading conversations...</div>
        </div>
    `;
    
    const mainContent = document.querySelector('.main-content');
    mainContent.innerHTML = '';
    mainContent.appendChild(spinner);
    
    await fetchConversations();
    renderConversations(currentFilter);
}

// Audio playback functions
async function initializeAudioContext() {
    if (!audioContext) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }
    return audioContext;
}

function handleAudioChunk(data) {
    console.log('Received audio chunk:', data.chunk_num, 'for generation:', data.gen_id);
    
    if (data.gen_id !== currentAudioGenerationId) {
        // New audio generation, reset chunks
        audioChunks = [];
        currentAudioGenerationId = data.gen_id;
        console.log('Starting new audio generation:', data.gen_id);
    }
    
    try {
        // Convert array back to Float32Array
        const floatArray = new Float32Array(data.audio);
        audioChunks.push(floatArray);
        
        console.log(`Added chunk ${data.chunk_num}, total chunks: ${audioChunks.length}, total samples: ${floatArray.length}`);
        
        // Update model status to show audio is streaming
        const modelStatusEl = document.getElementById('modelStatus');
        if (modelStatusEl) {
            modelStatusEl.textContent = `Streaming audio... (${data.chunk_num} chunks)`;
            modelStatusEl.style.color = '#3b82f6'; // Blue for streaming
        }
        
        // If this is the first chunk, start playing after a short delay
        if (data.chunk_num === 1) {
            console.log('First audio chunk received, will start playback after buffer...');
            setTimeout(() => {
                if (audioChunks.length > 0 && !isPlayingAudio) {
                    playBufferedAudio();
                }
            }, 300);
        }
        
        // If we're already playing, add to buffer and continue
        if (isPlayingAudio && audioChunks.length > 1) {
            console.log('Adding chunk to ongoing playback');
        }
        
    } catch (error) {
        console.error('Error processing audio chunk:', error);
        showNotification('Error processing audio chunk', 'error');
    }
}

async function playBufferedAudio() {
    if (audioChunks.length === 0 || isPlayingAudio) {
        return;
    }
    
    isPlayingAudio = true;
    
    try {
        await initializeAudioContext();
        
        // Combine all chunks
        const totalLength = audioChunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const combinedAudio = new Float32Array(totalLength);
        
        let offset = 0;
        for (const chunk of audioChunks) {
            combinedAudio.set(chunk, offset);
            offset += chunk.length;
        }
        
        console.log(`Playing audio: ${combinedAudio.length} samples at 24000 Hz`);
        
        // Create audio buffer (assuming 24kHz sample rate from your generator)
        const audioBuffer = audioContext.createBuffer(1, combinedAudio.length, 24000);
        audioBuffer.copyToChannel(combinedAudio, 0);
        
        // Create source and play
        currentAudioSource = audioContext.createBufferSource();
        currentAudioSource.buffer = audioBuffer;
        currentAudioSource.connect(audioContext.destination);
        
        // Set up event handlers
        currentAudioSource.onended = () => {
            console.log('Audio playback finished');
            isPlayingAudio = false;
            currentAudioSource = null;
            
            // Update model status
            const modelStatusEl = document.getElementById('modelStatus');
            if (modelStatusEl) {
                modelStatusEl.textContent = 'All models loaded';
                modelStatusEl.style.color = '#10b981';
            }
            
            showNotification('Audio playback complete', 'success');
        };
        
        currentAudioSource.onerror = (error) => {
            console.error('Audio playback error:', error);
            isPlayingAudio = false;
            currentAudioSource = null;
            showNotification('Audio playback failed', 'error');
        };
        
        // Start playback
        console.log('Starting audio playback...');
        currentAudioSource.start();
        
        showNotification('Playing audio response...', 'info');
        
    } catch (error) {
        console.error('Error playing audio:', error);
        isPlayingAudio = false;
        showNotification('Error playing audio: ' + error.message, 'error');
    }
}

function stopAudioPlayback() {
    if (currentAudioSource) {
        try {
            currentAudioSource.stop();
            console.log('Audio playback stopped');
        } catch (e) {
            console.log('Audio source already stopped');
        }
        currentAudioSource = null;
    }
    isPlayingAudio = false;
    audioChunks = [];
    currentAudioGenerationId = null;
}

function renderConversations(filter = currentFilter) {
    const mainContent = document.querySelector('.main-content');
    mainContent.innerHTML = '';
    
    let filteredConvs = conversations;
    if (filter === 'starred') {
        filteredConvs = conversations.filter(c => c.starred);
    } else if (filter === 'recent') {
        const now = new Date();
        const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
        filteredConvs = conversations.filter(c => new Date(c.date) > yesterday);
    }
    
    if (filteredConvs.length === 0) {
        let message = '';
        let emoji = '';
        switch(filter) {
            case 'starred':
                message = 'No starred conversations yet';
                emoji = '⭐';
                break;
            case 'recent':
                message = 'No recent conversations';
                emoji = '🕒';
                break;
            default:
                message = 'No conversations yet';
                emoji = '💬';
        }
        
        const emptyState = document.createElement('div');
        emptyState.style.cssText = `
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 300px;
            text-align: center;
            color: #6b7280;
        `;
        emptyState.innerHTML = `
            <div style="font-size: 4rem; margin-bottom: 1rem;">${emoji}</div>
            <div style="font-size: 1.25rem; font-weight: 500; margin-bottom: 0.5rem;">${message}</div>
            <div style="font-size: 0.875rem;">Start a new conversation to see it here</div>
            <button id="refreshConversationsBtn" style="margin-top: 1rem; padding: 0.5rem 1rem; background-color: #4f46e5; color: white; border: none; border-radius: 0.375rem; cursor: pointer; transition: background-color 0.2s;">
                Refresh Conversations
            </button>
        `;
        mainContent.appendChild(emptyState);
        
        const refreshBtn = document.getElementById('refreshConversationsBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', refreshConversations);
        }
        return;
    }
    
    // Sort conversations by date (newest first)
    filteredConvs.sort((a, b) => new Date(b.date) - new Date(a.date));
    
    filteredConvs.forEach(conv => {
        const fullUser = escapeHtml(conv.user_message);
        const fullAi = escapeHtml(conv.ai_message);
        const previewUser = escapeHtml(getPreviewText(conv.user_message));
        const previewAi = escapeHtml(getPreviewText(conv.ai_message));
        
        const card = document.createElement('div');
        card.className = 'conversation-card';
        card.dataset.fullUser = fullUser;
        card.dataset.fullAi = fullAi;
        card.dataset.previewUser = previewUser;
        card.dataset.previewAi = previewAi;
        card.dataset.id = conv.id;
        
        const isStarred = conv.starred;
        
        card.innerHTML = `
            <div class="flex justify-between items-start mb-4">
                <div class="text-sm text-gray-500">${formatDate(conv.date)}</div>
                <div class="text-xs bg-gray-100 px-2 py-1 rounded">Conversation #${conv.id}</div>
            </div>
            <div class="conversation-sections">
                <div class="message-section user-section">
                    <div class="message-header">
                        <div class="icon-container user-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" class="icon-svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd" />
                            </svg>
                        </div>
                        <div class="text-blue-700 font-semibold message-label">User:</div>
                    </div>
                    <div class="message-content">
                        <div class="message-text">${previewUser}</div>
                    </div>
                </div>
                <div class="message-section ai-section">
                    <div class="message-header">
                        <div class="icon-container ai-icon">
                            <svg xmlns="http://www.w3.org/2000/svg" class="icon-svg" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd" />
                            </svg>
                        </div>
                        <div class="text-green-700 font-semibold message-label">AI Companion:</div>
                    </div>
                    <div class="message-content">
                        <div class="message-text">${previewAi}</div>
                    </div>
                </div>
            </div>
            <div class="action-icons">
                <div class="action-icon star-icon ${isStarred ? 'active' : ''}" data-id="${conv.id}">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                        <path fill-rule="evenodd" d="M10.788 3.21c.448-1.077 1.976-1.077 2.424 0l2.082 5.007 5.404.433c1.164.093 1.636 1.545.749 2.305l-4.117 3.527 1.257 5.273c.271 1.136-.964 2.033-1.96 1.425L12 18.354 7.373 21.18c-.996.608-2.231-.29-1.96-1.425l1.257-5.273-4.117-3.527c-.887-.76-.415-2.212.749-2.305l5.404-.433 2.082-5.006z" clip-rule="evenodd" />
                    </svg>
                </div>
                <div class="action-icon expand-icon" data-id="${conv.id}">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor">
                        <path fill-rule="evenodd" d="M15 3.75a.75.75 0 01.75-.75h4.5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0V5.56l-3.97 3.97a.75.75 0 11-1.06-1.06l3.97-3.97h-2.69a.75.75 0 01-.75-.75zm-12 0A.75.75 0 013.75 3h4.5a.75.75 0 010 1.5H5.56l3.97 3.97a.75.75 0 01-1.06 1.06L4.5 5.56v2.69a.75.75 0 01-1.5 0v-4.5zm11.47 14.78a.75.75 0 111.06-1.06l3.97 3.97v-2.69a.75.75 0 011.5 0v4.5a.75.75 0 01-1.5 0v-4.5a.75.75 0 010-1.5h4.5a.75.75 0 010 1.5h-2.69l-3.97-3.97zm-4.94-1.06a.75.75 0 010 1.06L5.56 19.5h2.69a.75.75 0 010 1.5h-4.5a.75.75 0 01-.75-.75v-4.5a.75.75 0 011.5 0v2.69l3.97-3.97a.75.75 0 011.06 0z" clip-rule="evenodd" />
                    </svg>
                </div>
            </div>
        `;
        
        mainContent.appendChild(card);
    });
    
    // Add event listeners for star icons
    document.querySelectorAll('.star-icon').forEach(icon => {
        icon.addEventListener('click', function() {
            const id = parseInt(this.getAttribute('data-id'), 10);
            const conv = conversations.find(c => c.id === id);
            if (conv) {
                conv.starred = !conv.starred;
                this.classList.toggle('active');
            }
        });
    });
    
    // Add event listeners for expand icons
    document.querySelectorAll('.expand-icon').forEach(icon => {
        icon.addEventListener('click', function() {
            const id = this.getAttribute('data-id');
            const card = this.closest('.conversation-card');
            const isExpanded = card.classList.contains('expanded');
            
            card.classList.toggle('expanded');
            const userContent = card.querySelector('.user-section .message-text');
            const aiContent = card.querySelector('.ai-section .message-text');
            
            if (!isExpanded) {
                userContent.innerHTML = card.dataset.fullUser;
                aiContent.innerHTML = card.dataset.fullAi;
                this.querySelector('svg').innerHTML = `<path fill-rule="evenodd" d="M4.5 12a.75.75 0 01.75-.75h13.5a.75.75 0 010 1.5H5.25a.75.75 0 01-.75-.75z" clip-rule="evenodd" />`;
            } else {
                userContent.innerHTML = card.dataset.previewUser;
                aiContent.innerHTML = card.dataset.previewAi;
                this.querySelector('svg').innerHTML = `<path fill-rule="evenodd" d="M15 3.75a.75.75 0 01.75-.75h4.5a.75.75 0 01.75.75v4.5a.75.75 0 01-1.5 0V5.56l-3.97 3.97a.75.75 0 11-1.06-1.06l3.97-3.97h-2.69a.75.75 0 01-.75-.75zm-12 0A.75.75 0 013.75 3h4.5a.75.75 0 010 1.5H5.56l3.97 3.97a.75.75 0 01-1.06 1.06L4.5 5.56v2.69a.75.75 0 01-1.5 0v-4.5zm11.47 14.78a.75.75 0 111.06-1.06l3.97 3.97v-2.69a.75.75 0 011.5 0v4.5a.75.75 0 01-1.5 0v-4.5a.75.75 0 010-1.5h4.5a.75.75 0 010 1.5h-2.69l-3.97-3.97zm-4.94-1.06a.75.75 0 010 1.06L5.56 19.5h2.69a.75.75 0 010 1.5h-4.5a.75.75 0 01-.75-.75v-4.5a.75.75 0 011.5 0v2.69l3.97-3.97a.75.75 0 011.06 0z" clip-rule="evenodd" />`;
            }
        });
    });
}

// Model Status Management
let modelStatus = 'loading'; // loading, connected, disconnected
let ws = null;
let statusCheckInterval = null;

function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
}

function updateConnectionStatus() {
    const statusEl = document.getElementById('connectionStatus');
    const modelStatusEl = document.getElementById('modelStatus');
    const userEmailEl = document.getElementById('currentUserEmail');
    
    if (!statusEl || !modelStatusEl || !userEmailEl) return;
    
    switch(modelStatus) {
        case 'connected':
            statusEl.textContent = 'Connected';
            statusEl.style.color = STATUS_COLORS.connected.status;
            userEmailEl.style.color = STATUS_COLORS.connected.user;
            modelStatusEl.textContent = 'Ready';
            modelStatusEl.style.color = STATUS_COLORS.connected.status;
            break;
        case 'loading':
            statusEl.textContent = 'Connecting';
            statusEl.style.color = STATUS_COLORS.loading.status;
            userEmailEl.style.color = STATUS_COLORS.loading.user;
            modelStatusEl.textContent = 'Loading...';
            modelStatusEl.style.color = STATUS_COLORS.loading.status;
            break;
        case 'disconnected':
            statusEl.textContent = 'Disconnected';
            statusEl.style.color = STATUS_COLORS.disconnected.status;
            userEmailEl.style.color = STATUS_COLORS.disconnected.user;
            modelStatusEl.textContent = 'Offline';
            modelStatusEl.style.color = STATUS_COLORS.disconnected.status;
            break;
        default:
            statusEl.textContent = 'Connecting';
            statusEl.style.color = STATUS_COLORS.loading.status;
            userEmailEl.style.color = STATUS_COLORS.loading.user;
            modelStatusEl.textContent = 'Checking...';
            modelStatusEl.style.color = STATUS_COLORS.loading.status;
    }
}

async function checkModelStatus() {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        
        const data = await response.json();
        
        if (data.models_loaded) {
            modelStatus = 'connected';
        } else if (data.whisper_loaded || data.llm_loaded || data.rag_loaded) {
            modelStatus = 'connected';
        } else {
            modelStatus = 'disconnected';
        }
        
        updateConnectionStatus();
        updatePulseAnimation();
        
        return data;
    } catch (error) {
        console.error('Error checking model status:', error);
        modelStatus = 'disconnected';
        updateConnectionStatus();
        updatePulseAnimation();
        return null;
    }
}

function updatePulseAnimation() {
    const pulseContainer = document.querySelector('.pulse-container');
    const dotsPulse = document.querySelector('.dots-pulse');
    
    if (!pulseContainer || !dotsPulse) return;
    
    pulseContainer.classList.remove('connected', 'loading', 'disconnected');
    dotsPulse.classList.remove('connected', 'loading', 'disconnected');
    
    pulseContainer.classList.add(modelStatus);
    dotsPulse.classList.add(modelStatus);
}

// Notification system
function showNotification(message, type = 'info') {
    const colors = {
        info: '#3b82f6',
        success: '#10b981',
        warning: '#f59e0b',
        error: '#ef4444'
    };
    
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${colors[type] || colors.info};
        color: white;
        padding: 12px 18px;
        border-radius: 6px;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        animation: slideIn 0.3s ease;
    `;
    
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

function setupWebSocket() {
    const sessionToken = getCookie('session_token');
    const wsUrl = sessionToken 
        ? `ws://${window.location.host}/ws?session_token=${encodeURIComponent(sessionToken)}`
        : `ws://${window.location.host}/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        modelStatus = 'loading';
        updateConnectionStatus();
        
        checkModelStatus();
        if (statusCheckInterval) clearInterval(statusCheckInterval);
        statusCheckInterval = setInterval(checkModelStatus, 10000);
        
        ws.send(JSON.stringify({ type: 'test', message: 'Connection test' }));
        ws.send(JSON.stringify({ type: 'request_conversation_history' }));
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        modelStatus = 'disconnected';
        updateConnectionStatus();
        updatePulseAnimation();
        
        setTimeout(() => {
            console.log('Attempting to reconnect...');
            setupWebSocket();
        }, 5000);
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        modelStatus = 'disconnected';
        updateConnectionStatus();
        updatePulseAnimation();
    };
    
    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log('WebSocket message received:', data.type);
            
            switch(data.type) {
                case 'test_response':
                    console.log('Server test response:', data.message);
                    if (data.user_email && data.user_email !== 'anonymous') {
                        const emailEl = document.getElementById('currentUserEmail');
                        if (emailEl) {
                            emailEl.textContent = data.user_email;
                            updateConnectionStatus();
                        }
                    }
                    break;
                    
                case 'connection_established':
                    console.log('Connection established with session:', data.session_id);
                    if (data.user_email && data.user_email !== 'anonymous') {
                        const emailEl = document.getElementById('currentUserEmail');
                        if (emailEl) {
                            emailEl.textContent = data.user_email;
                            updateConnectionStatus();
                        }
                    }
                    break;
                    
                case 'audio_chunk':
                    handleAudioChunk(data);
                    break;
                    
                case 'audio_status':
                    handleAudioStatus(data);
                    break;
                    
                case 'response':
                    handleAIResponse(data);
                    break;
                    
                case 'conversation_history':
                    updateConversationsFromServer(data.conversations);
                    break;
                    
                case 'error':
                    console.error('Server error:', data.message);
                    modelStatus = 'disconnected';
                    updateConnectionStatus();
                    updatePulseAnimation();
                    break;
            }
        } catch (error) {
            console.error('Error parsing WebSocket message:', error);
        }
    };
}

function handleAudioStatus(data) {
    console.log('Audio status:', data.status, 'Gen ID:', data.gen_id);
    
    const modelStatusEl = document.getElementById('modelStatus');
    
    switch(data.status) {
        case 'generating':
            showNotification('Generating audio response...', 'info');
            if (modelStatusEl) {
                modelStatusEl.textContent = 'Generating audio...';
                modelStatusEl.style.color = '#f59e0b';
            }
            break;
        case 'first_chunk':
            showNotification('Audio generation started', 'info');
            if (modelStatusEl) {
                modelStatusEl.textContent = 'Audio streaming...';
                modelStatusEl.style.color = '#3b82f6';
            }
            break;
        case 'complete':
            showNotification('Audio generation complete', 'success');
            if (modelStatusEl) {
                modelStatusEl.textContent = 'Audio complete';
                modelStatusEl.style.color = '#10b981';
            }
            
            // If we have chunks but haven't started playing, start now
            setTimeout(() => {
                if (audioChunks.length > 0 && !isPlayingAudio) {
                    console.log('Starting delayed playback after completion');
                    playBufferedAudio();
                }
            }, 500);
            break;
        case 'interrupted':
            stopAudioPlayback();
            showNotification('Audio generation interrupted', 'warning');
            if (modelStatusEl) {
                modelStatusEl.textContent = 'Audio interrupted';
                modelStatusEl.style.color = '#ef4444';
            }
            break;
        case 'preparing_generation':
            showNotification('Preparing audio generation...', 'info');
            break;
    }
}

// Function to update conversations from server data
function updateConversationsFromServer(serverConversations) {
    if (!Array.isArray(serverConversations)) return;
    
    const newConversations = serverConversations.map(conv => ({
        id: conv.id,
        date: conv.timestamp,
        user_message: conv.user_message || '',
        ai_message: conv.ai_message || '',
        starred: conversations.find(c => c.server_id === conv.id)?.starred || false,
        audio_path: conv.audio_path || '',
        server_id: conv.id
    }));
    
    conversations = newConversations;
    conversationsLastUpdated = new Date();
    
    console.log(`Updated ${conversations.length} conversations from server`);
    renderConversations(currentFilter);
}

function handleAIResponse(data) {
    const newConversation = {
        id: conversations.length > 0 ? Math.max(...conversations.map(c => c.id)) + 1 : 1,
        date: new Date().toISOString().replace('T', ' ').substring(0, 19),
        user_message: window.lastUserMessage || 'User message',
        ai_message: data.text,
        starred: false,
        audio_path: data.audio_path || '',
        server_id: null
    };
    
    conversations.unshift(newConversation);
    window.lastUserMessage = null;
    renderConversations(currentFilter);
    
    setTimeout(() => {
        fetchConversations().then(() => {
            renderConversations(currentFilter);
        });
    }, 2000);
}

// Text input handling
function setupTextInput() {
    const textInput = document.getElementById('textInput');
    const sendBtn = document.getElementById('sendTextBtn');
    const charCount = document.getElementById('charCount');
    
    if (!textInput || !sendBtn || !charCount) return;
    
    textInput.addEventListener('input', () => {
        const length = textInput.value.length;
        charCount.textContent = `${length}/500`;
        
        sendBtn.disabled = length === 0 || length > 500 || modelStatus !== 'connected';
        
        if (length > 450) {
            charCount.style.color = '#ef4444';
        } else if (length > 400) {
            charCount.style.color = '#f59e0b';
        } else {
            charCount.style.color = '#6b7280';
        }
    });
    
    textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled && ws && ws.readyState === WebSocket.OPEN) {
                sendTextMessage();
            }
        } else if (e.key === 'Escape') {
            sendInterrupt();
        }
    });
    
    sendBtn.addEventListener('click', () => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            sendTextMessage();
        }
    });
    
    function sendTextMessage() {
        const message = textInput.value.trim();
        if (message && ws && ws.readyState === WebSocket.OPEN && modelStatus === 'connected') {
            window.lastUserMessage = message;
            
            ws.send(JSON.stringify({
                type: 'text_message',
                text: message,
                timestamp: new Date().toISOString()
            }));
            
            textInput.value = '';
            charCount.textContent = '0/500';
            sendBtn.disabled = true;
            charCount.style.color = '#6b7280';
            
            showNotification('Sending message...', 'info');
        }
    }
    
    function sendInterrupt() {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({
                type: 'interrupt'
            }));
            stopAudioPlayback();
            showNotification('Interrupt sent', 'warning');
        }
    }
}

// Auto-refresh conversations
function startAutoRefresh() {
    setInterval(async () => {
        if (document.visibilityState === 'visible') {
            await fetchConversations();
            renderConversations(currentFilter);
        }
    }, 30000);
}

// Initialize everything
document.addEventListener('DOMContentLoaded', async () => {
    // Add notification styles
    const notificationStyles = document.createElement('style');
    notificationStyles.textContent = `
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        @keyframes slideOut {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(notificationStyles);
    
    // Initialize filter buttons
    const filterButtons = document.querySelectorAll('.filter-btn');
    filterButtons.forEach(btn => {
        const filter = btn.getAttribute('data-filter');
        if (filter === currentFilter) {
            btn.classList.add('active');
        }
    });
    
    filterButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            filterButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentFilter = btn.getAttribute('data-filter');
            localStorage.setItem('conversationFilter', currentFilter);
            renderConversations(currentFilter);
        });
    });
    
    // Initial render with loading state
    const mainContent = document.querySelector('.main-content');
    mainContent.innerHTML = `
        <div class="conversation-card" style="text-align: center; padding: 40px;">
            <div class="loading-spinner" style="margin: 0 auto 20px; width: 40px; height: 40px; border: 4px solid #f3f4f6; border-top: 4px solid #4f46e5; border-radius: 50%; animation: spin 1s linear infinite;"></div>
            <div style="color: #6b7280; font-size: 1rem;">Loading conversations...</div>
        </div>
    `;
    
    // Set initial status
    modelStatus = 'loading';
    updateConnectionStatus();
    updatePulseAnimation();
    
    // Update username placeholder
    const userEmailEl = document.getElementById('currentUserEmail');
    if (userEmailEl && (!userEmailEl.textContent || userEmailEl.textContent.trim() === '')) {
        userEmailEl.textContent = 'loading';
        userEmailEl.style.color = STATUS_COLORS.loading.user;
    }
    
    // Load conversations from server
    await fetchConversations();
    
    // Initial render with loaded conversations
    renderConversations(currentFilter);
    
    // Setup WebSocket and status checking
    setupWebSocket();
    
    // Setup text input
    setupTextInput();
    
    // Check model status immediately
    checkModelStatus();
    
    // Start auto-refresh
    startAutoRefresh();
    
    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        if (ws) {
            ws.close();
        }
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        stopAudioPlayback();
    });
});