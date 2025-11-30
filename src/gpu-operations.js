export class GPUOperations {
    constructor(engine) {
        this.engine = engine;
    }

    async add(a, b) { return this.engine.runOp('op_add', a, b); }
    async sub(a, b) { return this.engine.runOp('op_sub', a, b); }
    async mul(a, b) { return this.engine.runOp('op_mul', a, b); }
    async div(a, b) { return this.engine.runOp('op_div', a, b); }
    async sqrt(a) { return this.engine.runOp('op_sqrt', a); }
    async sin(a) { return this.engine.runOp('op_trig', a, null, [0n], this.engine.constants.two_pi); }
    async cos(a) { return this.engine.runOp('op_trig', a, null, [1n], this.engine.constants.two_pi); }
}
