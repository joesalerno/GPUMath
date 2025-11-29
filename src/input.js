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

        // Current Mouse Position relative to center of screen (in pixels)
        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const px = mx - this.canvas.width / 2;
        const py = my - this.canvas.height / 2;

        const oldScale = this.app.camera.scale;
        let newScale = oldScale;

        // Zoom logic
        if (e.deltaY < 0) {
            newScale = (oldScale * 9n) / 10n; // Zoom In
        } else {
            newScale = (oldScale * 11n) / 10n; // Zoom Out
        }

        // Prevent scale from becoming 0
        if (newScale === 0n) newScale = 1n;

        // Adjust Center so that the point under the mouse remains stable
        // Old World Point = Center + P * OldScale
        // New World Point = NewCenter + P * NewScale
        // We want Old World Point == New World Point
        // Center + P * OldScale = NewCenter + P * NewScale
        // NewCenter = Center + P * (OldScale - NewScale)

        const diffScale = oldScale - newScale;
        const offsetX = BigInt(Math.round(px)) * diffScale;
        const offsetY = BigInt(Math.round(py)) * diffScale;

        this.app.camera.x += offsetX;
        this.app.camera.y += offsetY;
        this.app.camera.scale = newScale;
    }

    onResize() {
        this.app.resize();
    }
}
