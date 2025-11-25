class GPUBigNum {
  constructor(limbCount = 64) { // 64 * 32-bit = 2048-bit precision
    this.L = limbCount;
    this.device = null;
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter.requestDevice();
  }

  // Helper: JS BigInt -> Uint32Array (Little Endian)
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

  // Helper: Uint32Array -> JS BigInt
  fromBuffer(arr) {
    const res = [];
    const count = arr.length / this.L;
    for (let i = 0; i < count; i++) {
      let n = 0n;
      for (let j = this.L - 1; j >= 0; j--) {
        n = (n << 32n) | BigInt(arr[i * this.L + j]);
      }
      res.push(n);
    }
    return res;
  }

  async batchAdd(listA, listB) {
    const count = listA.length;
    const arrayA = this.toBuffer(listA);
    const arrayB = this.toBuffer(listB);
    const bufSize = arrayA.byteLength;

    // 1. Create Buffers
    const gpuBufferA = this._createBuf(arrayA, GPUBufferUsage.STORAGE);
    const gpuBufferB = this._createBuf(arrayB, GPUBufferUsage.STORAGE);
    const gpuResult  = this._createBuf(bufSize, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const gpuRead    = this._createBuf(bufSize, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);

    // 2. The "Awesome" Compute Shader
    const shader = `
      struct BigInt { limbs: array<u32, ${this.L}> }
      struct Data { values: array<BigInt> }
      
      @group(0) @binding(0) var<storage, read> bufA : Data;
      @group(0) @binding(1) var<storage, read> bufB : Data;
      @group(0) @binding(2) var<storage, read_write> bufR : Data;

      @compute @workgroup_size(64)
      fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
        let idx = global_id.x;
        if (idx >= arrayLength(&bufA.values)) { return; }

        var carry : u32 = 0u;
        // Unrolled loop for performance on compiled pipeline
        for (var i = 0u; i < ${this.L}u; i++) {
          let a = bufA.values[idx].limbs[i];
          let b = bufB.values[idx].limbs[i];
          let sum = a + b + carry;
          
          // Determine overflow (new carry)
          // If sum < a, we wrapped around. Or if carry was 1 and sum == a.
          carry = u32(sum < a || (carry == 1u && sum == a));
          
          bufR.values[idx].limbs[i] = sum;
        }
      }
    `;

    // 3. Pipeline Setup
    const module = this.device.createShaderModule({ code: shader });
    const pipeline = this.device.createComputePipeline({
      layout: 'auto', compute: { module, entryPoint: 'main' }
    });

    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: gpuBufferA } },
        { binding: 1, resource: { buffer: gpuBufferB } },
        { binding: 2, resource: { buffer: gpuResult } }
      ]
    });

    // 4. Dispatch
    const cmdEncoder = this.device.createCommandEncoder();
    const pass = cmdEncoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(count / 64));
    pass.end();

    // Copy result to read buffer
    cmdEncoder.copyBufferToBuffer(gpuResult, 0, gpuRead, 0, bufSize);
    this.device.queue.submit([cmdEncoder.finish()]);

    // 5. Read back
    await gpuRead.mapAsync(GPUMapMode.READ);
    const resultData = new Uint32Array(gpuRead.getMappedRange());
    const finalBigInts = this.fromBuffer(resultData);
    gpuRead.unmap();
    
    return finalBigInts;
  }

  _createBuf(dataOrSize, usage) {
    const size = typeof dataOrSize === 'number' ? dataOrSize : dataOrSize.byteLength;
    const buffer = this.device.createBuffer({ size, usage, mappedAtCreation: typeof dataOrSize !== 'number' });
    if (typeof dataOrSize !== 'number') {
      new Uint32Array(buffer.getMappedRange()).set(dataOrSize);
      buffer.unmap();
    }
    return buffer;
  }
}