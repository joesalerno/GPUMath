export class InputHandler {
    constructor(app) {
        this.app = app;
        this.canvas = app.canvas;
        this.isDragging = false;
        this.lastMouse = { x: 0, y: 0 };

        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        window.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('wheel', this.onWheel.bind(this), { passive: false });
        window.addEventListener('resize', this.onResize.bind(this));
    }

    onMouseDown(e) {
        this.isDragging = true;
        this.lastMouse = { x: e.clientX, y: e.clientY };
    }

    onMouseUp() {
        this.isDragging = false;
    }

    onMouseMove(e) {
        if (!this.isDragging) return;
        const dx = e.clientX - this.lastMouse.x;
        const dy = e.clientY - this.lastMouse.y;
        this.lastMouse = { x: e.clientX, y: e.clientY };
        this.app.camera.x += BigInt(-dx) * this.app.camera.scale;
        this.app.camera.y += BigInt(-dy) * this.app.camera.scale;
    }

    onWheel(e) {
        e.preventDefault();
        const zoomFactor = (e.deltaY < 0) ? 10n / 11n : 11n / 10n;
        this.app.camera.scale *= zoomFactor;
    }

    onResize() {
        this.app.resize();
    }
}
