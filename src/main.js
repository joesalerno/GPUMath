import { GPUEngine } from './gpu-engine.js';

class App {
    constructor() {
        this.canvas = document.getElementById('gpuCanvas');
        this.context = this.canvas.getContext('webgpu');
        this.logDiv = document.getElementById('log');

        this.engine = new GPUEngine(64, 32); // L=64, F=32
        this.camera = {
            x: 0n,
            y: 0n,
            scale: this.engine.floatToBig(4.0 / 800.0),
            resolution: new Float32Array([this.canvas.width, this.canvas.height])
        };

        this.ui = {
            cx: document.getElementById('cx'),
            cy: document.getElementById('cy'),
            zoom: document.getElementById('zoom'),
            fps: document.getElementById('fps')
        };
    }

    log(msg) {
        this.logDiv.innerText += `> ${msg}\n`;
        this.logDiv.scrollTop = this.logDiv.scrollHeight;
    }

    async init() {
        try {
            await this.engine.init();
            this.log("GPU Engine Ready.");
        } catch (e) {
            this.log("Error: " + e.message);
            return;
        }

        this.setupCanvas();
        this.setupUI();
        this.setupInputHandlers();
        this.createUBO();

        requestAnimationFrame(this.render.bind(this));
    }

    setupCanvas() {
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.engine.device, format });
        this.renderPipeline = this.engine.createRenderPipeline(format);
    }

    setupUI() {
        document.getElementById('btnReset').onclick = () => {
            this.camera.x = 0n;
            this.camera.y = 0n;
            this.camera.scale = this.engine.floatToBig(4.0 / 800.0);
        };
        document.getElementById('btnTests').onclick = () => this.runTests();
    }

    setupInputHandlers() {
        let isDragging = false;
        let lastMouse = { x: 0, y: 0 };

        this.canvas.addEventListener('mousedown', e => {
            isDragging = true;
            lastMouse = { x: e.clientX, y: e.clientY };
        });
        window.addEventListener('mouseup', () => { isDragging = false; });
        this.canvas.addEventListener('mousemove', e => {
            if (!isDragging) return;
            const dx = e.clientX - lastMouse.x;
            const dy = e.clientY - lastMouse.y;
            lastMouse = { x: e.clientX, y: e.clientY };
            this.camera.x += BigInt(-dx) * this.camera.scale;
            this.camera.y += BigInt(-dy) * this.camera.scale;
        });
        this.canvas.addEventListener('wheel', e => {
            e.preventDefault();
            this.camera.scale *= (e.deltaY < 0) ? 10n / 11n : 11n / 10n;
        }, { passive: false });

        window.addEventListener('resize', () => this.resize());
        this.resize();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        this.camera.resolution[0] = this.canvas.width;
        this.camera.resolution[1] = this.canvas.height;
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({ device: this.engine.device, format });
    }

    createUBO() {
        const UBO_SIZE = 256 + 256 + 256 + 16;
        this.uniformBuffer = this.engine.device.createBuffer({
            size: UBO_SIZE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.uboData = new Uint8Array(UBO_SIZE);
        this.bindGroup = this.engine.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }]
        });
    }

    updateUBO() {
        this.uboData.set(new Uint8Array(this.engine.toBuffer([this.camera.x]).buffer), 0);
        this.uboData.set(new Uint8Array(this.engine.toBuffer([this.camera.y]).buffer), 256);
        this.uboData.set(new Uint8Array(this.engine.toBuffer([this.camera.scale]).buffer), 512);
        new Float32Array(this.uboData.buffer, 768, 2).set(this.camera.resolution);
        this.engine.device.queue.writeBuffer(this.uniformBuffer, 0, this.uboData);

        this.ui.cx.innerText = this.engine.bigToFloatStr(this.camera.x);
        this.ui.cy.innerText = this.engine.bigToFloatStr(this.camera.y);
        this.ui.zoom.innerText = "Scale: " + this.engine.bigToFloatStr(this.camera.scale);
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
            this.ui.fps.innerText = `FPS: ${this.frameCount}`;
            this.frameCount = 0;
            this.lastFpsTime = time;
        }
    }

    async runTests() {
        this.log("Running Math Tests...");
        try {
            const A = this.engine.floatToBig(123.456);
            const B = this.engine.floatToBig(789.123);
            const negA = -A;

            const sum = await this.engine.add([A], [B]);
            this.log(`ADD: 123.456 + 789.123 = ${this.engine.bigToFloatStr(sum[0])}`);

            const mul = await this.engine.mul([A], [B]);
            this.log(`MUL: 123.456 * 789.123 = ${this.engine.bigToFloatStr(mul[0])}`);

            const mulNeg = await this.engine.mul([negA], [B]);
            this.log(`MUL NEG: -123.456 * 789.123 = ${this.engine.bigToFloatStr(mulNeg[0])}`);
            if (this.engine.bigToFloatStr(mulNeg[0]).startsWith("-97421")) this.log("✅ Signed Math OK");
            else this.log("❌ Signed Math Fail");

            const PI = this.engine.floatToBig(3.14159265);
            const sinPI = await this.engine.sin([PI]);
            this.log(`SIN(PI): ${this.engine.bigToFloatStr(sinPI[0])} (Exp ~0)`);
        } catch (e) {
            this.log("Test Error: " + e);
            console.error(e);
        }
    }
}

const app = new App();
app.init();
