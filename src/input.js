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
        this.app.isDirty = true;
    }

    onWheel(e) {
        e.preventDefault();

        const rect = this.canvas.getBoundingClientRect();
        const mx = e.clientX - rect.left;
        const my = e.clientY - rect.top;
        const px = mx - this.canvas.width / 2;
        const py = my - this.canvas.height / 2;

        const oldScale = this.app.camera.scale;
        let newScale = oldScale;

        if (e.deltaY < 0) {
            newScale = (oldScale * 9n) / 10n; // Zoom In
        } else {
            newScale = (oldScale * 11n) / 10n; // Zoom Out
        }

        if (newScale === 0n) newScale = 1n;

        const diffScale = oldScale - newScale;

        // Improved precision: px is float, diffScale is fixed point BigInt.
        // We multiply px by a large factor (e.g., 256) to keep sub-pixel precision,
        // multiply by diffScale, then divide by the factor.
        // Since diffScale is already scaled by F, we don't need to shift it further,
        // just handle the scalar multiplication carefully.

        const K = 1000n;
        const pxBig = BigInt(Math.round(px * Number(K)));
        const pyBig = BigInt(Math.round(py * Number(K)));

        const offsetX = (pxBig * diffScale) / K;
        const offsetY = (pyBig * diffScale) / K;

        this.app.camera.x += offsetX;
        this.app.camera.y += offsetY;
        this.app.camera.scale = newScale;
        this.app.isDirty = true;
    }

    onResize() {
        this.app.resize();
    }
}
