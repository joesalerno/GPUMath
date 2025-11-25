export class UI {
    constructor(app) {
        this.app = app;
        this.elements = {
            log: document.getElementById('log'),
            cx: document.getElementById('cx'),
            cy: document.getElementById('cy'),
            zoom: document.getElementById('zoom'),
            fps: document.getElementById('fps'),
        };
        this.setupButtons();
    }

    setupButtons() {
        document.getElementById('btnReset').onclick = () => this.app.resetView();
        document.getElementById('btnTests').onclick = () => this.app.runTests();
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
