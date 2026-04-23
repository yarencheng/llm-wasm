import { LlmInference, FilesetResolver } from '@mediapipe/tasks-genai';

const statusEl = document.getElementById('status');
const statusText = statusEl.querySelector('.status-text');
const welcomeScreen = document.getElementById('welcome-screen');
const progressContainer = document.getElementById('loading-progress-container');
const progressBar = document.getElementById('loading-progress');
const progressText = document.getElementById('progress-text');
const messagesList = document.getElementById('messages');
const promptInput = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');

let llmInference;

// 1. Initialize the LLM Inference Task
async function initLLM() {
    const modelUrl = 'https://pub-d5d788cf21574eb7951b65a7a4f469ac.r2.dev/gemma-3n-E2B-it-int4-Web.litertlm';
    
    try {
        progressContainer.classList.remove('hidden');
        statusText.textContent = 'Downloading model...';

        // Manual fetch to track progress
        const response = await fetch(modelUrl);
        const reader = response.body.getReader();
        const contentLength = +response.headers.get('Content-Length');
        
        let receivedLength = 0;
        let chunks = []; 
        
        while(true) {
            const {done, value} = await reader.read();
            if (done) break;
            
            chunks.push(value);
            receivedLength += value.length;
            
            // Update progress bar
            if (contentLength) {
                const percent = Math.round((receivedLength / contentLength) * 100);
                progressBar.style.width = `${percent}%`;
                progressText.textContent = `Downloading: ${percent}% (${Math.round(receivedLength/1024/1024)}MB / ${Math.round(contentLength/1024/1024)}MB)`;
            } else {
                progressText.textContent = `Downloading: ${Math.round(receivedLength/1024/1024)}MB`;
            }
        }

        statusText.textContent = 'Initializing engine...';
        progressText.textContent = 'Loading weights into GPU memory...';
        
        // Setup WASM fileset
        const genai = await FilesetResolver.forGenAiTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@latest/wasm"
        );

        // Combine chunks into a single Uint8Array
        const modelBuffer = new Uint8Array(receivedLength);
        let position = 0;
        for(let chunk of chunks) {
            modelBuffer.set(chunk, position);
            position += chunk.length;
        }

        // Initialize inference engine using the downloaded buffer
        llmInference = await LlmInference.createFromOptions(genai, {
            baseOptions: {
                modelAssetBuffer: modelBuffer
            },
            maxTokens: 1024,
            topK: 40,
            temperature: 0.8,
            randomSeed: 42
        });

        // Update UI to ready state
        statusEl.classList.remove('loading');
        statusEl.classList.add('ready');
        statusText.textContent = 'Ready';
        
        progressContainer.classList.add('hidden');
        promptInput.disabled = false;
        sendBtn.disabled = false;
        promptInput.placeholder = "Message Gemma 3...";
        
        console.log("LLM Initialized successfully");

    } catch (error) {
        console.error("Failed to initialize LLM:", error);
        statusText.textContent = 'Error loading model';
        progressText.textContent = 'Error: ' + error.message;
        progressText.style.color = '#ff5252';
    }
}

// 2. Handle Message Sending
async function sendMessage() {
    const text = promptInput.value.trim();
    if (!text || !llmInference) return;

    // Clear welcome screen on first message
    if (welcomeScreen) {
        welcomeScreen.remove();
    }

    // Add user message
    addMessage(text, 'user');
    promptInput.value = '';
    promptInput.disabled = true;
    sendBtn.disabled = true;

    // Add assistant message placeholder
    const assistantMsgEl = addMessage('', 'assistant');
    let fullResponse = '';

    try {
        // Stream response
        await llmInference.generateResponse(text, (partialResult, done) => {
            fullResponse += partialResult;
            assistantMsgEl.textContent = fullResponse;
            // Scroll to bottom
            messagesList.scrollTop = messagesList.scrollHeight;

            if (done) {
                promptInput.disabled = false;
                sendBtn.disabled = false;
                promptInput.focus();
            }
        });
    } catch (error) {
        console.error("Inference error:", error);
        assistantMsgEl.textContent = "Error: " + error.message;
        promptInput.disabled = false;
        sendBtn.disabled = false;
    }
}

// Helper: Add message to UI
function addMessage(text, role) {
    const div = document.createElement('div');
    div.className = `message ${role}`;
    div.textContent = text;
    messagesList.appendChild(div);
    messagesList.scrollTop = messagesList.scrollHeight;
    return div;
}

// Event Listeners
sendBtn.addEventListener('click', sendMessage);
promptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Start initialization
initLLM();