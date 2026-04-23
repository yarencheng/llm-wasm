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
    try {
        progressContainer.classList.remove('hidden');

        // Setup WASM fileset
        const genai = await FilesetResolver.forGenAiTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@latest/wasm"
        );

        statusText.textContent = 'Loading model...';

        // Initialize inference engine
        llmInference = await LlmInference.createFromOptions(genai, {
            baseOptions: {
                modelAssetPath: '/gemma-3n-E2B-it-int4-Web.litertlm'
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