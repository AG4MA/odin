/**
 * ODIN WASM Runtime
 * =================
 * JavaScript API per inference ODIN nel browser.
 */

// Types
interface OdinConfig {
  modelPath: string;
  wasmPath?: string;
  numThreads?: number;
  maxSequenceLength?: number;
}

interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  topP?: number;
  stopSequences?: string[];
}

interface TokenStream {
  [Symbol.asyncIterator](): AsyncIterator<string>;
}

// Main Runtime Class
class OdinRuntime {
  private config: OdinConfig;
  private session: any = null;
  private tokenizer: any = null;
  private isLoaded: boolean = false;
  private wasmModule: any = null;

  constructor(config: OdinConfig) {
    this.config = {
      numThreads: navigator.hardwareConcurrency || 4,
      maxSequenceLength: 2048,
      ...config
    };
  }

  /**
   * Load model and initialize runtime
   */
  async load(onProgress?: (progress: number) => void): Promise<void> {
    console.log('[ODIN] Loading model...');
    
    // 1. Load WASM runtime
    if (this.config.wasmPath) {
      await this.loadWasm(this.config.wasmPath);
    }
    
    // 2. Load ONNX model
    await this.loadOnnxModel(this.config.modelPath, onProgress);
    
    // 3. Initialize tokenizer
    await this.initTokenizer();
    
    this.isLoaded = true;
    console.log('[ODIN] Model loaded successfully');
  }

  private async loadWasm(wasmPath: string): Promise<void> {
    // Load custom WASM kernels
    const response = await fetch(wasmPath);
    const wasmBuffer = await response.arrayBuffer();
    const wasmModule = await WebAssembly.instantiate(wasmBuffer, {
      env: {
        memory: new WebAssembly.Memory({ initial: 256, maximum: 512 })
      }
    });
    this.wasmModule = wasmModule.instance.exports;
  }

  private async loadOnnxModel(modelPath: string, onProgress?: (p: number) => void): Promise<void> {
    // Use ONNX Runtime Web
    // @ts-ignore
    const ort = await import('onnxruntime-web');
    
    // Configure for performance
    ort.env.wasm.numThreads = this.config.numThreads!;
    ort.env.wasm.simd = true;
    
    // Load model with progress tracking
    const response = await fetch(modelPath);
    const contentLength = response.headers.get('content-length');
    const total = parseInt(contentLength || '0', 10);
    
    let loaded = 0;
    const reader = response.body!.getReader();
    const chunks: Uint8Array[] = [];
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      onProgress?.(loaded / total);
    }
    
    const modelBuffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      modelBuffer.set(chunk, offset);
      offset += chunk.length;
    }
    
    this.session = await ort.InferenceSession.create(modelBuffer.buffer);
  }

  private async initTokenizer(): Promise<void> {
    // Simple character-level tokenizer (replace with BPE in production)
    this.tokenizer = {
      encode: (text: string): number[] => {
        const ids = [2]; // BOS
        for (const char of text) {
          ids.push(char.charCodeAt(0) + 4);
        }
        return ids;
      },
      decode: (ids: number[]): string => {
        return ids
          .filter(id => id > 3)
          .map(id => String.fromCharCode(id - 4))
          .join('');
      }
    };
  }

  /**
   * Generate text from prompt
   */
  async generate(prompt: string, options: GenerateOptions = {}): Promise<string> {
    if (!this.isLoaded) {
      throw new Error('Model not loaded. Call load() first.');
    }

    const {
      maxTokens = 256,
      temperature = 0.8,
      topK = 40,
    } = options;

    let inputIds = this.tokenizer.encode(prompt);
    const generatedIds: number[] = [];

    for (let i = 0; i < maxTokens; i++) {
      // Run inference
      const logits = await this.forward(inputIds);
      
      // Sample next token
      const nextToken = this.sampleToken(logits, temperature, topK);
      
      // Check for EOS
      if (nextToken === 3) break;
      
      generatedIds.push(nextToken);
      inputIds = [nextToken]; // For RNN-style models, only need last token
    }

    return this.tokenizer.decode(generatedIds);
  }

  /**
   * Stream generation token by token
   */
  async *generateStream(prompt: string, options: GenerateOptions = {}): AsyncGenerator<string> {
    if (!this.isLoaded) {
      throw new Error('Model not loaded. Call load() first.');
    }

    const { maxTokens = 256, temperature = 0.8, topK = 40 } = options;

    let inputIds = this.tokenizer.encode(prompt);

    for (let i = 0; i < maxTokens; i++) {
      const logits = await this.forward(inputIds);
      const nextToken = this.sampleToken(logits, temperature, topK);
      
      if (nextToken === 3) break;
      
      yield this.tokenizer.decode([nextToken]);
      inputIds = [nextToken];
    }
  }

  private async forward(inputIds: number[]): Promise<Float32Array> {
    // @ts-ignore
    const ort = await import('onnxruntime-web');
    
    const inputTensor = new ort.Tensor(
      'int64',
      BigInt64Array.from(inputIds.map(BigInt)),
      [1, inputIds.length]
    );

    const outputs = await this.session.run({ input_ids: inputTensor });
    const logits = outputs.logits.data as Float32Array;
    
    // Return logits for last token
    const vocabSize = 32768;
    const lastTokenLogits = new Float32Array(vocabSize);
    const offset = (inputIds.length - 1) * vocabSize;
    
    for (let i = 0; i < vocabSize; i++) {
      lastTokenLogits[i] = logits[offset + i];
    }
    
    return lastTokenLogits;
  }

  private sampleToken(logits: Float32Array, temperature: number, topK: number): number {
    // Apply temperature
    for (let i = 0; i < logits.length; i++) {
      logits[i] /= temperature;
    }

    // Top-K filtering
    const indices = Array.from({ length: logits.length }, (_, i) => i);
    indices.sort((a, b) => logits[b] - logits[a]);
    
    const topKIndices = indices.slice(0, topK);
    const topKLogits = topKIndices.map(i => logits[i]);
    
    // Softmax
    const maxLogit = Math.max(...topKLogits);
    const expLogits = topKLogits.map(l => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map(e => e / sumExp);
    
    // Sample
    const r = Math.random();
    let cumsum = 0;
    for (let i = 0; i < probs.length; i++) {
      cumsum += probs[i];
      if (r < cumsum) {
        return topKIndices[i];
      }
    }
    
    return topKIndices[topKIndices.length - 1];
  }

  /**
   * Get model info
   */
  getInfo(): object {
    return {
      loaded: this.isLoaded,
      config: this.config,
      inputNames: this.session?.inputNames,
      outputNames: this.session?.outputNames
    };
  }

  /**
   * Cleanup resources
   */
  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    this.isLoaded = false;
  }
}

// Factory function
async function createOdinRuntime(config: OdinConfig): Promise<OdinRuntime> {
  const runtime = new OdinRuntime(config);
  return runtime;
}

// Export for module systems
export { OdinRuntime, createOdinRuntime };
export type { OdinConfig, GenerateOptions };
