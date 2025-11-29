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
        // Zoom logic using integer arithmetic to avoid zero result
        // Zoom In: scale * 0.9 (approx) -> scale * 9 / 10
        // Zoom Out: scale * 1.1 (approx) -> scale * 11 / 10
        if (e.deltaY < 0) {
            this.app.camera.scale = (this.app.camera.scale * 9n) / 10n;
        } else {
            this.app.camera.scale = (this.app.camera.scale * 11n) / 10n;
        }
    }

    onResize() {
        this.app.resize();
    }
}
