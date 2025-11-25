import { GPUEngine } from './gpu-engine.js';

// --- CONFIGURATION ---
const L = 64; // Limbs (64 * 32 = 2048 bits)
const F = 32; // Fractional Limbs (1024 bits)

// --- APPLICATION LOGIC ---
(async () => {
    const canvas = document.getElementById('gpuCanvas');
    const context = canvas.getContext('webgpu');
    const logDiv = document.getElementById('log');
    const log = (msg) => { logDiv.innerText += `> ${msg}\n`; logDiv.scrollTop = logDiv.scrollHeight; };

    const engine = new GPUEngine(L, F);
    try {
        await engine.init();
        log("GPU Engine Ready.");
    } catch (e) {
        log("Error: " + e.message);
        return;
    }

    // Setup Canvas Format
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: engine.device, format });
    const renderPipeline = engine.createRenderPipeline(format);

    // Camera State
    const camera = {
        x: 0n, // BigInt (Fixed Point)
        y: 0n,
        scale: engine.floatToBig(4.0 / 800.0), // Start showing ~4 units wide.
        resolution: new Float32Array([canvas.width, canvas.height])
    };

    // UI References
    const ui = {
        cx: document.getElementById('cx'),
        cy: document.getElementById('cy'),
        zoom: document.getElementById('zoom'),
        fps: document.getElementById('fps')
    };

    // Input Handling
    let isDragging = false;
    let lastMouse = { x: 0, y: 0 };

    canvas.addEventListener('mousedown', e => { isDragging = true; lastMouse = { x: e.clientX, y: e.clientY }; });
    window.addEventListener('mouseup', () => isDragging = false);

    canvas.addEventListener('mousemove', e => {
        if (!isDragging) return;
        const dx = e.clientX - lastMouse.x;
        const dy = e.clientY - lastMouse.y;
        lastMouse = { x: e.clientX, y: e.clientY };

        const bigDx = BigInt(-dx) * camera.scale;
        const bigDy = BigInt(-dy) * camera.scale;

        camera.x += bigDx;
        camera.y += bigDy;
    });

    canvas.addEventListener('wheel', e => {
        e.preventDefault();
        if (e.deltaY < 0) {
            camera.scale = camera.scale * 10n / 11n; // Zoom In
        } else {
            camera.scale = camera.scale * 11n / 10n; // Zoom Out
        }
    }, { passive: false });

    // Resize
    const resize = () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        camera.resolution[0] = canvas.width;
        camera.resolution[1] = canvas.height;
        context.configure({ device: engine.device, format });
    };
    window.addEventListener('resize', resize);
    resize();

    // Uniform Buffer
    const UBO_SIZE = 256 + 256 + 256 + 16;
    const uniformBuffer = engine.device.createBuffer({ size: UBO_SIZE, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const uboData = new Uint8Array(UBO_SIZE);

    const updateUBO = () => {
        const bx = new Uint32Array(engine.toBuffer([camera.x]).buffer);
        const by = new Uint32Array(engine.toBuffer([camera.y]).buffer);
        const bs = new Uint32Array(engine.toBuffer([camera.scale]).buffer);

        uboData.set(new Uint8Array(bx.buffer), 0);
        uboData.set(new Uint8Array(by.buffer), 256);
        uboData.set(new Uint8Array(bs.buffer), 512);

        new Float32Array(uboData.buffer, 768, 2).set(camera.resolution);

        engine.device.queue.writeBuffer(uniformBuffer, 0, uboData);

        ui.cx.innerText = engine.bigToFloatStr(camera.x);
        ui.cy.innerText = engine.bigToFloatStr(camera.y);
        ui.zoom.innerText = "Scale: " + engine.bigToFloatStr(camera.scale);
    };

    // Bind Group
    const bindGroup = engine.device.createBindGroup({
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }]
    });

    // Render Loop
    let lastTime = 0;
    let frameCount = 0;
    let lastFpsTime = 0;

    const render = (time) => {
        const dt = time - lastTime;
        lastTime = time;

        updateUBO();

        const encoder = engine.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0,0,0,1] }]
        });
        pass.setPipeline(renderPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(6);
        pass.end();
        engine.device.queue.submit([encoder.finish()]);

        frameCount++;
        if (time - lastFpsTime > 1000) {
            ui.fps.innerText = `FPS: ${frameCount}`;
            frameCount = 0;
            lastFpsTime = time;
        }

        requestAnimationFrame(render);
    };
    requestAnimationFrame(render);

    // --- Buttons ---
    document.getElementById('btnReset').onclick = () => {
        camera.x = 0n;
        camera.y = 0n;
        camera.scale = engine.floatToBig(4.0 / 800.0);
    };

    document.getElementById('btnTests').onclick = async () => {
        log("Running Math Tests...");
        try {
            const A = engine.floatToBig(123.456);
            const B = engine.floatToBig(789.123);

            // Add
            const sum = (await engine.runOp('add', [A], [B]))[0];
            log(`ADD: 123.456 + 789.123 = ${engine.bigToFloatStr(sum)}`);

            // Mul
            const mul = (await engine.runOp('mul', [A], [B]))[0];
            log(`MUL: 123.456 * 789.123 = ${engine.bigToFloatStr(mul)}`);

            // Signed Test
            const negA = -A;
            const mulNeg = (await engine.runOp('mul', [negA], [B]))[0];
            log(`MUL NEG: -123.456 * 789.123 = ${engine.bigToFloatStr(mulNeg)}`);

            if (engine.bigToFloatStr(mulNeg).startsWith("-97421")) log("✅ Signed Math OK");
            else log("❌ Signed Math Fail");

            // Trig Test (Sin PI)
            const PI = engine.floatToBig(3.14159265);
            const sinPI = (await engine.sin([PI]))[0];
            log(`SIN(PI): ${engine.bigToFloatStr(sinPI)} (Exp ~0)`);

        } catch(e) {
            log("Test Error: " + e);
            console.error(e);
        }
    };
})();
