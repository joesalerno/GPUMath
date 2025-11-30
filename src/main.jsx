import { GPUEngine } from './gpu-engine.js';
import { GPUOperations } from './gpu-operations.js';
import { UI } from './ui.js';
import { InputHandler } from './input.js';
import { runSuite } from './test-cases.js';

class App {
    constructor() {
        this.canvas = document.getElementById('gpuCanvas');
        this.context = this.canvas.getContext('webgpu');

        // Reduced precision to L=16 (512 bits) to prevent GPU hang (TDR).
        // F=8 provides 256 bits of fractional precision, leaving 256 bits for integer part.
        this.engine = new GPUEngine(16, 8);
        this.math = new GPUOperations(this.engine);
        this.ui = new UI(this);
        this.input = new InputHandler(this);

        this.camera = {
            x: 0n,
            y: 0n,
            scale: 0n, // Initialized in resetView
            resolution: new Float32Array([this.canvas.width, this.canvas.height]),
            maxIter: 255
        };
        this.resetView();
    }

    log(msg) {
        this.ui.log(msg);
    }

    async init() {
        try {
            const shaderCode = await fetch('/fractal.wgsl').then(res => res.text());
            await this.engine.init(shaderCode);
            this.log("GPU Engine Ready.");
        } catch (e) {
            this.log("Error: " + e.message);
            console.error(e);
            return;
        }

        this.setupCanvas();
        this.createUBO();
        this.resize();

        requestAnimationFrame(this.render.bind(this));
    }

    setupCanvas() {
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.engine.device, format });
        this.renderPipeline = this.engine.createRenderPipeline(format, 'vs_main', 'fs_main');
    }

    resetView() {
        this.camera.x = 0n;
        this.camera.y = 0n;
        this.camera.scale = this.engine.floatToBig(4.0 / this.canvas.width);
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.camera.resolution[0] = this.canvas.width;
        this.camera.resolution[1] = this.canvas.height;

        if (this.engine.device) {
            const format = navigator.gpu.getPreferredCanvasFormat();
            this.context.configure({ device: this.engine.device, format });
        }
    }

    createUBO() {
        const L_BYTES = this.engine.L * 4;
        // Layout: centerX(L), centerY(L), scale(L), resolution(2*4), maxIter(4), padding(4)
        // Note: Storage buffer layout.
        // We need to ensure alignment. LargeInt is array<u32, L>.
        // L is usually multiple of 2 or 4.

        this.uboOffsets = {
            x: 0,
            y: L_BYTES,
            scale: L_BYTES * 2,
            res: L_BYTES * 3,
            iter: L_BYTES * 3 + 8
        };

        const TOTAL_SIZE = L_BYTES * 3 + 16; // +16 covers vec2 + u32 + padding

        this.uniformBuffer = this.engine.device.createBuffer({
            size: TOTAL_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.uboData = new Uint8Array(TOTAL_SIZE);
        this.bindGroup = this.engine.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } }
            ]
        });
    }

    updateUBO() {
        this.uboData.set(new Uint8Array(this.engine.toBuffer([this.camera.x]).buffer), this.uboOffsets.x);
        this.uboData.set(new Uint8Array(this.engine.toBuffer([this.camera.y]).buffer), this.uboOffsets.y);
        this.uboData.set(new Uint8Array(this.engine.toBuffer([this.camera.scale]).buffer), this.uboOffsets.scale);
        new Float32Array(this.uboData.buffer, this.uboOffsets.res, 2).set(this.camera.resolution);
        new Uint32Array(this.uboData.buffer, this.uboOffsets.iter, 1)[0] = this.camera.maxIter;
        this.engine.device.queue.writeBuffer(this.uniformBuffer, 0, this.uboData);
        this.ui.updateCameraInfo();
    }

    render(time) {
        this.updateUBO();
        const encoder = this.engine.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: [0, 0, 0, 1]
            }]
        });
        pass.setPipeline(this.renderPipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6);
        pass.end();
        this.engine.device.queue.submit([encoder.finish()]);

        this.updateFPS(time);
        requestAnimationFrame(this.render.bind(this));
    }

    updateFPS(time) {
        if (!this.lastFpsTime) this.lastFpsTime = 0;
        if (!this.frameCount) this.frameCount = 0;
        this.frameCount++;
        if (time - this.lastFpsTime > 1000) {
            this.ui.updateFPS(this.frameCount);
            this.frameCount = 0;
            this.lastFpsTime = time;
        }
    }

    async runTests() {
        await runSuite(this.engine, this.math, this.log.bind(this));
    }
}

const app = new App();
app.init();
