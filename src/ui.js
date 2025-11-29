export class UI {
    constructor(app) {
        this.app = app;
        this.elements = {
            log: document.getElementById('log'),
            cx: document.getElementById('cx'),
            cy: document.getElementById('cy'),
            zoom: document.getElementById('zoom'),
            fps: document.getElementById('fps'),
            maxIterRange: document.getElementById('maxIterRange'),
            maxIterVal: document.getElementById('maxIterVal'),
        };
        this.setupButtons();
        this.setupControls();
    }

    setupButtons() {
        document.getElementById('btnReset').onclick = () => this.app.resetView();
        document.getElementById('btnTests').onclick = () => this.app.runTests();
    }

    setupControls() {
        this.elements.maxIterRange.oninput = (e) => {
            const val = parseInt(e.target.value);
            this.app.camera.maxIter = val;
            this.elements.maxIterVal.innerText = val;
        };
    }

    log(msg) {
        this.elements.log.innerText += `> ${msg}\n`;
        this.elements.log.scrollTop = this.elements.log.scrollHeight;
    }

    updateCameraInfo() {
        this.elements.cx.innerText = this.app.engine.bigToFloatStr(this.app.camera.x);
        this.elements.cy.innerText = this.app.engine.bigToFloatStr(this.app.camera.y);
        this.elements.zoom.innerText = "Scale: " + this.app.engine.bigToFloatStr(this.app.camera.scale);
    }

    updateFPS(fps) {
        this.elements.fps.innerText = `FPS: ${fps}`;
    }
}
