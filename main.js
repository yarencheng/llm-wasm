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
const clearCacheBtn = document.getElementById('clear-cache-btn');

let llmInference;
let modelStorage;

const MODEL_NAME = 'gemma-3n-E2B-it-int4';
const DB_NAME = 'LLMCache';
const STORE_NAME = 'chunks';
const META_STORE = 'metadata';

class ModelStorage {
    constructor() {
        this.db = null;
    }

    async init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, 1);
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains(STORE_NAME)) {
                    db.createObjectStore(STORE_NAME);
                }
                if (!db.objectStoreNames.contains(META_STORE)) {
                    db.createObjectStore(META_STORE);
                }
            };
            request.onsuccess = (e) => {
                this.db = e.target.result;
                resolve();
            };
            request.onerror = (e) => reject(e.target.error);
        });
    }

    async getMetadata() {
        return this._get(META_STORE, MODEL_NAME);
    }

    async setMetadata(meta) {
        return this._put(META_STORE, MODEL_NAME, meta);
    }

    async saveChunk(index, data) {
        return this._put(STORE_NAME, index, data);
    }

    async getChunkCount() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([STORE_NAME], 'readonly');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.count();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getAllChunks() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([STORE_NAME], 'readonly');
            const store = transaction.objectStore(STORE_NAME);
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async clear() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([STORE_NAME, META_STORE], 'readwrite');
            transaction.objectStore(STORE_NAME).clear();
            transaction.objectStore(META_STORE).clear();
            transaction.oncomplete = () => resolve();
            transaction.onerror = () => reject(transaction.error);
        });
    }

    _get(storeName, key) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([storeName], 'readonly');
            const store = transaction.objectStore(storeName);
            const request = store.get(key);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    _put(storeName, key, value) {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction([storeName], 'readwrite');
            const store = transaction.objectStore(storeName);
            const request = store.put(value, key);
            request.onsuccess = () => resolve();
            request.onerror = () => reject(request.error);
        });
    }
}

// 1. Initialize the LLM Inference Task
async function initLLM() {
    const modelUrl = 'https://pub-d5d788cf21574eb7951b65a7a4f469ac.r2.dev/gemma-3n-E2B-it-int4-Web.litertlm';
    
    try {
        modelStorage = new ModelStorage();
        await modelStorage.init();

        const metadata = await modelStorage.getMetadata();
        const chunkCount = await modelStorage.getChunkCount();
        
        progressContainer.classList.remove('hidden');
        
        let modelBlob;
        const headResponse = await fetch(modelUrl, { method: 'HEAD' });
        const currentETag = headResponse.headers.get('ETag');
        const totalSize = parseInt(headResponse.headers.get('Content-Length'));

        // If metadata matches and we have all chunks, load from cache
        if (metadata && metadata.etag === currentETag && metadata.totalSize === totalSize && metadata.complete) {
            statusText.textContent = 'Loading from cache...';
            const chunks = await modelStorage.getAllChunks();
            modelBlob = new Blob(chunks, { type: 'application/octet-stream' });
        } else {
            // Check if we can resume
            let startByte = 0;
            let chunks = [];
            
            if (metadata && metadata.etag === currentETag) {
                // Same file, try to resume
                const existingChunks = await modelStorage.getAllChunks();
                chunks = existingChunks;
                startByte = chunks.reduce((acc, chunk) => acc + chunk.byteLength, 0);
                statusText.textContent = startByte > 0 ? 'Resuming download...' : 'Downloading model...';
            } else {
                // Different file or no metadata, start fresh
                await modelStorage.clear();
                await modelStorage.setMetadata({ etag: currentETag, totalSize, complete: false });
            }

            const fetchOptions = startByte > 0 ? { headers: { 'Range': `bytes=${startByte}-` } } : {};
            const response = await fetch(modelUrl, fetchOptions);
            
            if (!response.ok && response.status !== 206) {
                throw new Error(`Failed to fetch model: ${response.statusText}`);
            }

            const reader = response.body.getReader();
            let receivedLength = startByte;
            
            while(true) {
                const {done, value} = await reader.read();
                if (done) break;
                
                // Save chunk to IDB
                const chunkIndex = chunks.length;
                chunks.push(value);
                await modelStorage.saveChunk(chunkIndex, value);
                
                receivedLength += value.length;
                
                // Update progress bar
                const percent = Math.round((receivedLength / totalSize) * 100);
                progressBar.style.width = `${percent}%`;
                progressText.textContent = `Downloading: ${percent}% (${Math.round(receivedLength/1024/1024)}MB / ${Math.round(totalSize/1024/1024)}MB)`;
            }

            await modelStorage.setMetadata({ etag: currentETag, totalSize, complete: true });
            modelBlob = new Blob(chunks, { type: 'application/octet-stream' });
        }

        statusText.textContent = 'Initializing engine...';
        progressText.textContent = 'Loading weights into GPU memory...';
        
        // Setup WASM fileset
        const genai = await FilesetResolver.forGenAiTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@latest/wasm"
        );

        // Initialize inference engine using the Blob URL (more memory efficient)
        const modelObjectURL = URL.createObjectURL(modelBlob);
        
        llmInference = await LlmInference.createFromOptions(genai, {
            baseOptions: {
                modelAssetPath: modelObjectURL
            },
            maxTokens: 1024,
            topK: 40,
            temperature: 0.8,
            randomSeed: 42
        });

        // Revoke the URL to free up memory once loaded
        URL.revokeObjectURL(modelObjectURL);

        // Update UI to ready state
        statusEl.classList.remove('loading');
        statusEl.classList.add('ready');
        statusText.textContent = 'Ready';
        
        progressContainer.classList.add('hidden');
        promptInput.disabled = false;
        sendBtn.disabled = false;
        promptInput.placeholder = "Message Gemma 3...";
        
        // Show clear cache button if model is cached
        if (metadata && metadata.complete) {
            clearCacheBtn.classList.remove('hidden');
        }
        
        console.log("LLM Initialized successfully");

    } catch (error) {
        console.error("Failed to initialize LLM:", error);
        statusText.textContent = 'Error loading model';
        progressText.textContent = 'Error: ' + error.message;
        progressText.style.color = '#ff5252';
        
        // If error, maybe clear storage to try again next time
        if (modelStorage) {
            // await modelStorage.clear(); // Careful here, maybe don't clear on every error
        }
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

clearCacheBtn.addEventListener('click', async () => {
    if (confirm('Are you sure you want to delete the cached model weights? You will need to download them again.')) {
        await modelStorage.clear();
        location.reload();
    }
});

// Start initialization
initLLM();