import { GPUEngine } from './gpu-engine.js';
import { GPUOperations } from './gpu-operations.js';
import { UI } from './ui.js';
import { InputHandler } from './input.js';
import { runSuite } from './test-cases.js';

class App {
    constructor() {
        this.canvas = document.getElementById('gpuCanvas');
        this.context = this.canvas.getContext('webgpu');

        this.engine = new GPUEngine(16, 8);
        this.math = new GPUOperations(this.engine);
        this.ui = new UI(this);
        this.input = new InputHandler(this);

        this.camera = {
            x: 0n,
            y: 0n,
            scale: 0n,
            resolution: new Float32Array([this.canvas.width, this.canvas.height]),
            maxIter: 50 // Reduced default for stability
        };
        this.isDirty = true; // render flag
        this.resetView();
    }

    log(msg) {
        this.ui.log(msg);
        console.log("[APP] " + msg);
    }

    async init() {
        if (!navigator.gpu) {
            this.log("WebGPU not supported by browser.");
            return;
        }

        try {
            const shaderCode = await fetch('/fractal.wgsl').then(res => res.text());
            await this.engine.init(shaderCode);
            this.log("GPU Engine Ready.");
        } catch (e) {
            this.log("Error: " + e.message);
            console.error(e);
            return;
        }

        // Hook device loss after init
        if (this.engine.device) {
             this.engine.device.lost.then((info) => {
                this.log(`GPU Device Lost: ${info.message}`);
                console.error("GPU Device Lost", info);
            });
        }

        this.setupCanvas();
        this.createUBO();
        this.resize();

        requestAnimationFrame(this.render.bind(this));
    }

    setupCanvas() {
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.engine.device,
            format,
            alphaMode: 'opaque' // Explicitly set opaque
        });
        this.renderPipeline = this.engine.createRenderPipeline(format, 'vs_main', 'fs_main');
    }

    resetView() {
        this.camera.x = 0n;
        this.camera.y = 0n;
        this.camera.scale = this.engine.floatToBig(4.0 / this.canvas.width);
        this.camera.maxIter = 50; // Ensure reset also respects stability
        this.isDirty = true;
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.camera.resolution[0] = this.canvas.width;
        this.camera.resolution[1] = this.canvas.height;
        this.isDirty = true;

        if (this.engine.device) {
            const format = navigator.gpu.getPreferredCanvasFormat();
            this.context.configure({
                device: this.engine.device,
                format,
                alphaMode: 'opaque'
            });
        }
    }

    createUBO() {
        const L_BYTES = this.engine.L * 4;
        this.uboOffsets = {
            x: 0,
            y: L_BYTES,
            scale: L_BYTES * 2,
            res: L_BYTES * 3,
            iter: L_BYTES * 3 + 8
        };

        const TOTAL_SIZE = L_BYTES * 3 + 16;

        this.uniformBuffer = this.engine.device.createBuffer({
            size: TOTAL_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.uboData = new Uint8Array(TOTAL_SIZE);

        // Views for faster setting
        this.uboResView = new Float32Array(this.uboData.buffer, this.uboOffsets.res, 2);
        this.uboIterView = new Uint32Array(this.uboData.buffer, this.uboOffsets.iter, 1);

        this.bindGroup = this.engine.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuffer } }
            ]
        });
    }

    updateUBO() {
        const bx = this.engine.toBuffer([this.camera.x]);
        this.uboData.set(new Uint8Array(bx.buffer), this.uboOffsets.x);

        const by = this.engine.toBuffer([this.camera.y]);
        this.uboData.set(new Uint8Array(by.buffer), this.uboOffsets.y);

        const bs = this.engine.toBuffer([this.camera.scale]);
        this.uboData.set(new Uint8Array(bs.buffer), this.uboOffsets.scale);

        this.uboResView.set(this.camera.resolution);
        this.uboIterView[0] = this.camera.maxIter;

        this.engine.device.queue.writeBuffer(this.uniformBuffer, 0, this.uboData);
        this.ui.updateCameraInfo();
        this.isDirty = false;
    }

    render(time) {
        if (this.isDirty) {
            this.updateUBO();
        }

        const encoder = this.engine.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: [0.1, 0.1, 0.1, 1] // Dark Grey
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
