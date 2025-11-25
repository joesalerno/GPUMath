import * as math from './math-utils.js';

export class GPUEngine {
    constructor(L, F) {
        this.L = L;
        this.F = F;
        this.device = null;
        this.shaderModule = null;
        this.pipelines = {};
        this.constants = {};
    }

    async init(shaderCode) {
        if (!navigator.gpu) throw new Error("WebGPU not supported");
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) throw new Error("No GPU adapter found");
        this.device = await adapter.requestDevice();
        this.shaderModule = this.device.createShaderModule({ code: shaderCode });
        this._createConstants();
        console.log("GPU Engine Initialized");
    }

    createComputePipeline(entryPoint) {
        if (!this.pipelines[entryPoint]) {
            this.pipelines[entryPoint] = this.device.createComputePipeline({
                layout: 'auto',
                compute: {
                    module: this.shaderModule,
                    entryPoint,
                    constants: {
                        L: this.L,
                        F: this.F,
                    }
                }
            });
        }
    }

    _createConstants() {
        const TWO_PI = "6487ED5110B4611A62633145C06E0E68948127044533E63A0105DF531D89CD9128A57F477590822765A1523B06C758169135064731F29C35C7433877995643640F11C89874136C055F60B84D2B196C27F0922872A437C0994C3817F723223126848A183D5D7716944B8411D44686475C62281D6F2C33D14D89CD0627721535451D00B026859752D5D00B89C6D39E837D8D6228076635292415516053748259463991C6E6A2689240361245787680D311E6A1221430F7C2037953258A3668393526E3082989D22784566270E03C1A32766397FC30846503715C6C075D1C689849E94D414619379685954B469950796865074E182522770248430541E1837F359051680186591295320076214C236E0D2C76A288E8367F7D21E428C6466986693892801F41B68F807466540673059695655513A4997096696C7540D3708D64C7203780D774653697968525049964585354972410";
        this.constants.two_pi = this.createBuffer(math.hexToFixed(TWO_PI, this.L, this.F));
    }

    async runOp(op, listA, listB, listC = null, bufB_override = null) {
        const count = listA.length;
        const bufA = this.createBuffer(math.toBuffer(listA, this.L));
        const bufB = bufB_override ? bufB_override : (listB ? this.createBuffer(math.toBuffer(listB, this.L)) : this.createBuffer(new Uint32Array(this.L)));
        const bufC = listC ? this.createBuffer(math.toBuffer(listC, this.L)) : this.createBuffer(new Uint32Array(this.L));

        const outSize = count * this.L * 4;
        const bufR = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bufRead = this.device.createBuffer({ size: outSize, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

        this.createComputePipeline(op);

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
        const res = math.fromBuffer(new Uint32Array(bufRead.getMappedRange()), this.L);
        bufRead.unmap();
        return res;
    }

    createBuffer(data) {
        const buf = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
        new Uint32Array(buf.getMappedRange()).set(data);
        buf.unmap();
        return buf;
    }

    // --- RENDER API ---
    createRenderPipeline(format, entryPointVS, entryPointFS) {
        return this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: this.shaderModule,
                entryPoint: entryPointVS,
                constants: {
                    L: this.L,
                    F: this.F,
                }
            },
            fragment: {
                module: this.shaderModule,
                entryPoint: entryPointFS,
                targets: [{ format }],
                constants: {
                    L: this.L,
                    F: this.F,
                }
            },
            primitive: { topology: 'triangle-list' }
        });
    }

    // --- MATH HELPERS ---
    toBuffer(bigInts) { return math.toBuffer(bigInts, this.L); }
    fromBuffer(arr) { return math.fromBuffer(arr, this.L); }
    floatToBig(v) { return math.floatToBig(v, this.F); }
    bigToFloatStr(n) { return math.bigToFloatStr(n, this.F); }
}
