import { LlmInference, FilesetResolver } from '@mediapipe/tasks-genai';

const statusEl = document.getElementById('status');
const statusText = statusEl.querySelector('.status-text');
const welcomeScreen = document.getElementById('welcome-screen');
const progressContainer = document.getElementById('loading-progress-container');
const progressBar = document.getElementById('loading-progress');
const progressText = document.getElementById('progress-text');
const clearCacheBtn = document.getElementById('clear-cache-btn');
const messagesList = document.getElementById('messages');
const promptInput = document.getElementById('prompt-input');
const sendBtn = document.getElementById('send-btn');

let llmInference;
let modelStorage;

class ModelStorage {
    constructor() {
        this.dbName = 'LLMStorage';
        this.dbVersion = 1;
        this.db = null;
    }

    init() {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(this.dbName, this.dbVersion);
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve();
            };
            request.onupgradeneeded = (e) => {
                const db = e.target.result;
                if (!db.objectStoreNames.contains('metadata')) {
                    db.createObjectStore('metadata');
                }
                if (!db.objectStoreNames.contains('chunks')) {
                    db.createObjectStore('chunks');
                }
            };
        });
    }

    async getMetadata() {
        return this._get('metadata', 'model_info');
    }

    async setMetadata(data) {
        return this._put('metadata', 'model_info', data);
    }

    async saveChunk(index, data) {
        return this._put('chunks', index, data);
    }

    async getChunkCount() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['chunks'], 'readonly');
            const store = transaction.objectStore('chunks');
            const request = store.count();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getAllChunks() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['chunks'], 'readonly');
            const store = transaction.objectStore('chunks');
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async clear() {
        return new Promise((resolve, reject) => {
            const transaction = this.db.transaction(['metadata', 'chunks'], 'readwrite');
            transaction.objectStore('metadata').clear();
            transaction.objectStore('chunks').clear();
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
        
        progressContainer.classList.remove('hidden');
        
        let chunks = [];
        let receivedLength = 0;
        
        let headResponse;
        let currentETag = null;
        let totalSize = 0;
        let corsError = false;

        try {
            headResponse = await fetch(modelUrl, { method: 'HEAD' });
            if (!headResponse.ok) {
                corsError = true;
            } else {
                currentETag = headResponse.headers.get('ETag');
                totalSize = parseInt(headResponse.headers.get('Content-Length'));
            }
        } catch (e) {
            console.warn("HEAD request failed, likely due to CORS. Falling back to simple GET.", e);
            corsError = true;
        }

        // If metadata matches and we have completed download, load from cache
        if (!corsError && metadata && metadata.etag === currentETag && metadata.totalSize === totalSize && metadata.complete) {
            statusText.textContent = 'Loading from cache...';
            chunks = await modelStorage.getAllChunks();
            receivedLength = totalSize;
            
            progressBar.style.width = `100%`;
            progressText.textContent = `Loaded from cache: (${Math.round(totalSize/1024/1024)}MB)`;
        } else {
            // Check if we can resume
            if (!corsError && metadata && metadata.etag === currentETag) {
                // Same file, try to resume
                chunks = await modelStorage.getAllChunks();
                receivedLength = chunks.reduce((acc, chunk) => acc + chunk.byteLength, 0);
                statusText.textContent = receivedLength > 0 ? 'Resuming download...' : 'Downloading model...';
            } else {
                // Different file or no metadata, start fresh
                await modelStorage.clear();
                if (!corsError) {
                    await modelStorage.setMetadata({ etag: currentETag, totalSize, complete: false });
                }
                statusText.textContent = 'Downloading model...';
                receivedLength = 0;
                chunks = [];
            }

            if (corsError || receivedLength < totalSize) {
                const fetchOptions = receivedLength > 0 && !corsError ? { headers: { 'Range': `bytes=${receivedLength}-` } } : {};
                let response;
                
                try {
                    response = await fetch(modelUrl, fetchOptions);
                } catch (e) {
                    if (receivedLength > 0) {
                        console.warn("Range request failed, likely due to CORS. Starting fresh.", e);
                        // Fallback: clear cache and restart without Range
                        receivedLength = 0;
                        chunks = [];
                        await modelStorage.clear();
                        response = await fetch(modelUrl);
                        corsError = true;
                    } else {
                        throw e;
                    }
                }
                
                if (!response.ok && response.status !== 206) {
                    throw new Error(`Failed to fetch model: ${response.statusText}`);
                }

                if (corsError) {
                    // Try to get total size from GET response if HEAD failed
                    totalSize = parseInt(response.headers.get('Content-Length')) || 0;
                }

                const reader = response.body.getReader();
                
                while(true) {
                    const {done, value} = await reader.read();
                    if (done) break;
                    
                    // Save chunk to IDB
                    const chunkIndex = chunks.length;
                    chunks.push(value);
                    if (!corsError) {
                        await modelStorage.saveChunk(chunkIndex, value);
                    }
                    
                    receivedLength += value.length;
                    
                    // Update progress bar
                    if (totalSize) {
                        const percent = Math.round((receivedLength / totalSize) * 100);
                        progressBar.style.width = `${percent}%`;
                        progressText.textContent = `Downloading: ${percent}% (${Math.round(receivedLength/1024/1024)}MB / ${Math.round(totalSize/1024/1024)}MB)`;
                    } else {
                        progressText.textContent = `Downloading: ${Math.round(receivedLength/1024/1024)}MB`;
                    }
                }

                if (!corsError && currentETag && totalSize > 0) {
                    await modelStorage.setMetadata({ etag: currentETag, totalSize, complete: true });
                }
            } else if (!corsError && receivedLength === totalSize) {
                 await modelStorage.setMetadata({ etag: currentETag, totalSize, complete: true });
            }
        }

        statusText.textContent = 'Initializing engine...';
        progressText.textContent = 'Loading weights into GPU memory...';
        
        // Setup WASM fileset
        const genai = await FilesetResolver.forGenAiTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-genai@latest/wasm"
        );

        // Create a Blob from the downloaded chunks to prevent OOM errors
        // Allocating a 3GB contiguous Uint8Array in JS often fails.
        const modelBlob = new Blob(chunks, { type: 'application/octet-stream' });
        const modelObjectURL = URL.createObjectURL(modelBlob);
        
        // Free up the JS array of chunks to save memory
        chunks = [];

        // Initialize inference engine using the downloaded blob URL
        llmInference = await LlmInference.createFromOptions(genai, {
            baseOptions: {
                modelAssetPath: modelObjectURL
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
        
        // Show clear cache button if model is cached
        if (clearCacheBtn) {
            clearCacheBtn.classList.remove('hidden');
        }
        
        console.log("LLM Initialized successfully");

    } catch (error) {
        console.error("Failed to initialize LLM:", error);
        statusText.textContent = 'Error loading model';
        progressText.textContent = 'Error: ' + error.message;
        progressText.style.color = '#ff5252';
        
        // If clear cache button exists, show it so user can reset if corrupted
        if (clearCacheBtn) {
            clearCacheBtn.classList.remove('hidden');
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

if (clearCacheBtn) {
    clearCacheBtn.addEventListener('click', async () => {
        if (confirm('Are you sure you want to delete the cached model weights? You will need to download them again.')) {
            if (modelStorage) {
                await modelStorage.clear();
            }
            location.reload();
        }
    });
}

// Start initialization
initLLM();