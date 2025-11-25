class GPUBigNum {
  constructor(limbCount = 64) {
    this.L = limbCount;
    this.device = null;
    this.pipelines = {}; // Cache for Add/Sub pipelines
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();

    // Compile the shader module ONCE
    const shaderCode = `
      struct BigInt { limbs: array<u32, ${this.L}> }
      struct Data { values: array<BigInt> }
      
      @group(0) @binding(0) var<storage, read> bufA : Data;
      @group(0) @binding(1) var<storage, read> bufB : Data;
      @group(0) @binding(2) var<storage, read_write> bufR : Data;

      // --- ADDITION KERNEL ---
      @compute @workgroup_size(64)
      fn main_add(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x;
        if (idx >= arrayLength(&bufA.values)) { return; }

        var carry = 0u;
        for (var i = 0u; i < ${this.L}u; i++) {
          let a = bufA.values[idx].limbs[i];
          let b = bufB.values[idx].limbs[i];
          let sum = a + b + carry;
          
          // Branchless Carry: 1 if overflow occurred, 0 otherwise
          carry = u32(sum < a || (carry == 1u && sum == a));
          
          bufR.values[idx].limbs[i] = sum;
        }
      }

      // --- SUBTRACTION KERNEL ---
      @compute @workgroup_size(64)
      fn main_sub(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x;
        if (idx >= arrayLength(&bufA.values)) { return; }

        var borrow = 0u;
        for (var i = 0u; i < ${this.L}u; i++) {
          let a = bufA.values[idx].limbs[i];
          let b = bufB.values[idx].limbs[i];
          
          // Calculate diff (wrapping is expected behavior in u32)
          let diff = a - b - borrow;

          // Branchless Borrow: If a < b, or if a == b and we already had a borrow
          // Logic: (a < b) OR ((a == b) AND borrow)
          let next_borrow = u32(a < b) | (u32(a == b) & borrow);
          
          bufR.values[idx].limbs[i] = diff;
          borrow = next_borrow;
        }
      }
    `;

    const module = this.device.createShaderModule({ code: shaderCode });

    // Create pipelines for both operations
    this.pipelines.add = this.device.createComputePipeline({
      layout: 'auto', compute: { module, entryPoint: 'main_add' }
    });
    this.pipelines.sub = this.device.createComputePipeline({
      layout: 'auto', compute: { module, entryPoint: 'main_sub' }
    });
  }

  // --- Public API ---

  async add(listA, listB) { return this._runOp(listA, listB, 'add'); }
  async sub(listA, listB) { return this._runOp(listA, listB, 'sub'); }

  // --- Internal Core ---

  async _runOp(listA, listB, opName) {
    const count = listA.length;
    // 1. Prepare Data
    const arrayA = this.toBuffer(listA);
    const arrayB = this.toBuffer(listB);
    const bufSize = arrayA.byteLength;

    // 2. GPU Memory Allocation
    // Note: MappedAtCreation=true allows us to write without a staging buffer copy
    const gpuA = this._createBuf(arrayA, GPUBufferUsage.STORAGE);
    const gpuB = this._createBuf(arrayB, GPUBufferUsage.STORAGE);
    const gpuR = this._createBuf(bufSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const gpuRead = this._createBuf(bufSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    // 3. Bind Resources
    const pipeline = this.pipelines[opName];
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuA } },
        { binding: 1, resource: { buffer: gpuB } },
        { binding: 2, resource: { buffer: gpuR } }
      ]
    });

    // 4. Execute
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();

    encoder.copyBufferToBuffer(gpuR, 0, gpuRead, 0, bufSize);
    this.device.queue.submit([encoder.finish()]);

    // 5. Read Results
    await gpuRead.mapAsync(GPUMapMode.READ);
    const resultBigInts = this.fromBuffer(new Uint32Array(gpuRead.getMappedRange()));
    gpuRead.unmap();

    return resultBigInts;
  }

  // --- Utilities ---

  _createBuf(dataOrSize, usage) {
    const size = typeof dataOrSize === 'number' ? dataOrSize : dataOrSize.byteLength;
    const buffer = this.device.createBuffer({ size, usage, mappedAtCreation: typeof dataOrSize !== 'number' });
    if (typeof dataOrSize !== 'number') {
      new Uint32Array(buffer.getMappedRange()).set(dataOrSize);
      buffer.unmap();
    }
    return buffer;
  }

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
    const count = arr.length / this.L;
    for (let i = 0; i < count; i++) {
      let n = 0n;
      for (let j = this.L - 1; j >= 0; j--) {
        // Treat as unsigned magnitude
        n = (n << 32n) | BigInt(arr[i * this.L + j]);
      }
      res.push(n);
    }
    return res;
  }
}