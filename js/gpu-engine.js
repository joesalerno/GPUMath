// --- GPU ENGINE CLASS ---
export class GPUEngine {
    constructor(L, F) {
        this.device = null;
        this.L = L;
        this.F = F;
        this.pipelines = {};
        this.constants = {};
    }

    async init() {
        if (!navigator.gpu) throw new Error("WebGPU not supported");
        const adapter = await navigator.gpu.requestAdapter();
        this.device = await adapter.requestDevice();

        let shaderCode = await fetch('wgsl/fractal.wgsl').then(res => res.text());
        shaderCode = shaderCode.replace(/override L: u32 = \d+u;/, `override L: u32 = ${this.L}u;`);
        shaderCode = shaderCode.replace(/override F: u32 = \d+u;/, `override F: u32 = ${this.F}u;`);
        this.shaderModule = this.device.createShaderModule({ code: shaderCode });

        // Create Compute Pipelines
        const ops = ['add', 'sub', 'mul', 'mul_int', 'div', 'mod', 'sqrt', 'exp', 'modpow', 'trig'];
        ops.forEach(op => {
            this.pipelines[op] = this.device.createComputePipeline({
                layout: 'auto',
                compute: { module: this.shaderModule, entryPoint: `op_${op}` }
            });
        });

        // Constants for Trig
        const TWO_PI = "6487ED5110B4611A62633145C06E0E68948127044533E63A0105DF531D89CD9128A57F477590822765A1523B06C758169135064731F29C35C7433877995643640F11C89874136C055F60B84D2B196C27F0922872A437C0994C3817F723223126848A183D5D7716944B8411D44686475C62281D6F2C33D14D89CD0627721535451D00B026859752D5D00B89C6D39E837D8D6228076635292415516053748259463991C6E6A2689240361245787680D311E6A1221430F7C2037953258A3668393526E3082989D22784566270E03C1A32766397FC30846503715C6C075D1C689849E94D414619379685954B469950796865074E182522770248430541E1837F359051680186591295320076214C236E0D2C76A288E8367F7D21E428C6466986693892801F41B68F807466540673059695655513A4997096696C7540D3708D64C7203780D774653697968525049964585354972410";
        this.constants.two_pi = this.createBuffer(this.hexToFixed(TWO_PI));

        console.log("GPU Engine Initialized");
    }

    // --- MATH API ---
    toBuffer(bigInts) {
        const arr = new Uint32Array(bigInts.length * this.L);
        bigInts.forEach((bn, i) => {
            let n = bn;
            for (let j = 0; j < this.L; j++) {
                arr[i * this.L + j] = Number(n & 0xFFFFFFFFn);
                n >>= 32n;
            }
        });
        return arr;
    }

    fromBuffer(arr) {
        const res = [];
        for (let i = 0; i < arr.length / this.L; i++) {
            let n = 0n;
            for (let j = this.L - 1; j >= 0; j--) {
                n = (n << 32n) | BigInt(arr[i * this.L + j]);
            }
            const msb = arr[i * this.L + this.L - 1] >>> 31;
            if (msb) {
                n = n - (1n << BigInt(this.L * 32));
            }
            res.push(n);
        }
        return res;
    }

    floatToBig(v) {
        const S = BigInt(this.F * 32);
        const val = BigInt(Math.round(v * Number(1n << 52n)));
        const shift = S - 52n;
        if (shift >= 0n) return val << shift;
        return val >> (-shift);
    }

    bigToFloatStr(n) {
        const S = BigInt(this.F * 32);
        const scale = 1n << S;
        let v = n;
        let sign = "";
        if (v < 0n) { sign = "-"; v = -v; }

        const intPart = v / scale;
        const fracPart = v % scale;
        const d = (fracPart * (10n ** 10n)) / scale;
        return `${sign}${intPart}.${d.toString().padStart(10, '0')}`;
    }

    hexToFixed(hex) { return this.toBuffer([BigInt("0x"+hex) << BigInt(this.F*32)]); }

    async runOp(op, listA, listB, listC = null, bufB_Override=null, bufC_Override=null) {
        const count = listA.length;
        const bufA = this.createBuffer(this.toBuffer(listA));

        let bufB = bufB_Override;
        if (!bufB) {
            if (listB) bufB = this.createBuffer(this.toBuffer(listB));
            else bufB = this.createBuffer(new Uint32Array(64));
        }

        let bufC = bufC_Override;
        if (!bufC) {
            if (listC) bufC = this.createBuffer(this.toBuffer(listC));
            else bufC = this.createBuffer(new Uint32Array(64));
        }

        const outSize = count * this.L * 4;
        const bufR = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bufRead = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

        const bindGroup = this.device.createBindGroup({
            layout: this.pipelines[op].getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: bufA } },
                { binding: 1, resource: { buffer: bufB } },
                { binding: 2, resource: { buffer: bufR } },
                { binding: 3, resource: { buffer: bufC } }
            ]
        });

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines[op]);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(count / 64));
        pass.end();
        encoder.copyBufferToBuffer(bufR, 0, bufRead, 0, outSize);
        this.device.queue.submit([encoder.finish()]);

        await bufRead.mapAsync(GPUMapMode.READ);
        const res = this.fromBuffer(new Uint32Array(bufRead.getMappedRange()));
        bufRead.unmap();
        return res;
    }

    createBuffer(data) {
        const buf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
        new Uint32Array(buf.getMappedRange()).set(data);
        buf.unmap();
        return buf;
    }

    // Helper wrappers
    async sin(a) { return this.runOp('trig', a, null, null, this.constants.two_pi, this.createBuffer(new Uint32Array(64).fill(0))); }
    async cos(a) { return this.runOp('trig', a, null, null, this.constants.two_pi, this.createBuffer(new Uint32Array(64).fill(1))); }

    // --- RENDER API ---
    createRenderPipeline(format) {
        return this.device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: this.shaderModule, entryPoint: 'vs_main' },
            fragment: { module: this.shaderModule, entryPoint: 'fs_main', targets: [{ format }] },
            primitive: { topology: 'triangle-list' }
        });
    }
}
