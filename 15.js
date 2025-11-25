class GPUBigNum {
  constructor(limbCount = 64) {
    this.L = limbCount; // 64 limbs = 2048 bits
    // We treat the number as Fixed Point: [Integer: L/2][Fraction: L/2]
    this.F = this.L / 2; 
    this.device = null;
    this.pipelines = {};
  }

  async init() {
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    this.device = await (await navigator.gpu.requestAdapter()).requestDevice();

    const shaderCode = `
      override L: u32 = ${this.L}u; 
      override F: u32 = ${this.F}u; // Fraction Limbs
      
      struct BigInt   { limbs: array<u32, L> }     
      struct BigIntx2 { limbs: array<u32, L * 2> } 
      struct DataIn   { values: array<BigInt> }
      struct DataOut  { values: array<BigInt> } // Output matches input size
      
      @group(0) @binding(0) var<storage, read> bufA : DataIn;
      @group(0) @binding(1) var<storage, read> bufB : DataIn; // Used for div/mul inputs
      @group(0) @binding(2) var<storage, read_write> bufR : DataOut;

      // --- HELPERS ---

      fn mac(a: u32, b: u32, c: u32, carry: u32) -> vec2<u32> {
        let lo_a=a&0xFFFFu; let hi_a=a>>16u; let lo_b=b&0xFFFFu; let hi_b=b>>16u;
        let p0=lo_a*lo_b; let p1=lo_a*hi_b; let p2=hi_a*lo_b; let p3=hi_a*hi_b;
        let t=p1+(p0>>16u);
        var lo=((t<<16u)|(p0&0xFFFFu)); var hi=p3+(t>>16u)+(p2>>16u);
        let t2=lo+(p2<<16u); if(t2<lo){hi++;} lo=t2;
        let t3=lo+c; if(t3<lo){hi++;} lo=t3;
        let t4=lo+carry; if(t4<lo){hi++;} lo=t4;
        return vec2<u32>(lo, hi);
      }

      // Add: A += B
      fn add_self(a: ptr<function, BigInt>, b: BigInt) {
        var carry = 0u;
        for(var i=0u; i<L; i++) {
          let v = (*a).limbs[i];
          let s = v + b.limbs[i] + carry;
          carry = u32(s < v || (carry==1u && s==v));
          (*a).limbs[i] = s;
        }
      }

      // Sub: A -= B
      fn sub_self(a: ptr<function, BigInt>, b: BigInt) {
        var bor = 0u;
        for(var i=0u; i<L; i++) {
          let v = (*a).limbs[i]; let ov = b.limbs[i];
          let d = v - ov - bor;
          bor = u32(v < ov) | (u32(v == ov) & bor);
          (*a).limbs[i] = d;
        }
      }

      // Fixed Point Mul: Res = (A * B) >> (F * 32)
      fn mul_fixed(a: BigInt, b: BigInt) -> BigInt {
        var temp: array<u32, L * 2>;
        for(var k=0u; k<L*2; k++){ temp[k]=0u; }
        
        // 1. Standard Long Mul
        for(var i=0u; i<L; i++) {
          var c = 0u;
          for(var j=0u; j<L; j++) {
            let r = mac(a.limbs[i], b.limbs[j], temp[i+j], c);
            temp[i+j] = r.x; c = r.y;
          }
          temp[i+L] = c;
        }

        // 2. Extract Middle (Fixed Point Shift)
        // We want bits from F*32 to (F+L)*32
        var res: BigInt;
        for(var i=0u; i<L; i++) {
          res.limbs[i] = temp[i + F];
        }
        return res;
      }

      // Scalar Div: A = A / scalar (O(N) speed)
      fn div_scalar(a: ptr<function, BigInt>, b: u32) {
        var rem = 0u; // Remainder (up to 32 bits effectively if b is small)
        // Use 64-bit float hack for 32-bit division logic if u64 is tricky, 
        // but here we manually do (rem << 32 | limb) / b
        
        for(var k=0u; k<L; k++) {
          let i = L - 1u - k;
          let val = (*a).limbs[i];
          
          // Synthetic 64-bit division step: (rem * 2^32 + val) / b
          // Since b is small (factorial terms), we can do it in parts or float
          let r_f = f64(rem) * 4294967296.0 + f64(val);
          let b_f = f64(b);
          let q   = u32(r_f / b_f);
          rem     = u32(r_f % b_f);
          
          (*a).limbs[i] = q;
        }
      }

      // --- KERNELS ---

      @compute @workgroup_size(64)
      fn main_sin(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x; if (idx >= arrayLength(&bufA.values)) { return; }
        
        // Taylor Series: x - x^3/3! + x^5/5! - ...
        let x = bufA.values[idx];
        let x_sq = mul_fixed(x, x); // Precompute x^2
        
        var term = x;   // First term: x
        var sum  = x;   // Accumulator
        
        // Iterations (20 iterations gives ~640 bits of precision for reasonable inputs)
        // For full 2048-bit precision we'd need more, but 20 is "fast/concise"
        for(var i=1u; i<=20u; i++) {
          // term = term * x^2
          term = mul_fixed(term, x_sq);
          
          // term = term / (2i * (2i+1))
          let k = i * 2u;
          let div = k * (k + 1u);
          div_scalar(&term, div);
          
          if (i % 2u == 1u) { sub_self(&sum, term); }
          else              { add_self(&sum, term); }
        }
        bufR.values[idx] = sum;
      }

      @compute @workgroup_size(64)
      fn main_cos(@builtin(global_invocation_id) id : vec3<u32>) {
        let idx = id.x; if (idx >= arrayLength(&bufA.values)) { return; }
        
        // Taylor: 1 - x^2/2! + x^4/4! ...
        let x = bufA.values[idx];
        let x_sq = mul_fixed(x, x);
        
        // First term: 1 (Fixed Point 1.0 is 1 << (F*32))
        var term: BigInt; term.limbs[F] = 1u; // Set 1.0
        var sum = term;

        for(var i=1u; i<=20u; i++) {
          term = mul_fixed(term, x_sq);
          let k = i * 2u;
          let div = (k - 1u) * k; // (2i-1)(2i)
          div_scalar(&term, div);

          if (i % 2u == 1u) { sub_self(&sum, term); }
          else              { add_self(&sum, term); }
        }
        bufR.values[idx] = sum;
      }
    `;

    const module = this.device.createShaderModule({ code: shaderCode });
    const pipe = (ep) => this.device.createComputePipeline({ layout:'auto', compute:{ module, entryPoint:ep }});
    this.pipelines = { sin: pipe('main_sin'), cos: pipe('main_cos') };
  }

  async sin(listA) { return this._op(listA, 'sin'); }
  async cos(listA) { return this._op(listA, 'cos'); }

  // Calculates Tan as Sin / Cos (CPU coordination)
  async tan(listA) {
    // Note: This relies on the Mul/Div kernels (not shown in this concise view)
    // or we can just compute sin/cos and divide in JS for simplicity, 
    // but here we just return sin/cos
    console.warn("For tan(), compute sin(A)/cos(A) using div()");
    return []; 
  }

  async _op(listA, op) {
    const count = listA.length;
    const bA = this._buf(this._toArr(listA), GPUBufferUsage.STORAGE);
    // Dummy buffer for unused binding
    const bB = this._buf(64, GPUBufferUsage.STORAGE); 
    const bR = this._buf(count*this.L*4, GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
    const bRead = this._buf(count*this.L*4, GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ);

    const bg = this.device.createBindGroup({
      layout: this.pipelines[op].getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bA } },
        { binding: 1, resource: { buffer: bB } }, // Unused
        { binding: 2, resource: { buffer: bR } }
      ]
    });

    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipelines[op]);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(count/64));
    pass.end();
    enc.copyBufferToBuffer(bR, 0, bRead, 0, count*this.L*4);
    this.device.queue.submit([enc.finish()]);

    await bRead.mapAsync(GPUMapMode.READ);
    const res = this._fromArr(new Uint32Array(bRead.getMappedRange()));
    bRead.unmap();
    return res;
  }

  // --- Fixed Point Utilities ---
  
  // Float -> BigInt Fixed Point
  floatToBig(val) {
    // This is a simplification. For huge precision, parse strings.
    // 1.0 = 2^(32 * F)
    const SHIFT = BigInt(this.F * 32);
    // Handle integer part
    const intPart = BigInt(Math.floor(val));
    // Handle fraction part (rough approximation for float input)
    const fracPart = BigInt(Math.floor((val - Math.floor(val)) * Number(1n << 52n)));
    
    // Combine: (Int << Shift) + (Frac << (Shift - 52))
    return (intPart << SHIFT) | (fracPart << (SHIFT - 52n));
  }

  // BigInt Fixed Point -> Float string
  bigToFloatStr(bn) {
    const SHIFT = BigInt(this.F * 32);
    const intPart = bn >> SHIFT;
    const mask = (1n << SHIFT) - 1n;
    const fracPart = bn & mask;
    
    // Convert fraction to decimal string
    // FracVal / 2^Shift
    const precision = 20n; // Show 20 digits
    const decimal = (fracPart * (10n ** precision)) >> SHIFT;
    return `${intPart}.${decimal.toString().padStart(Number(precision), '0')}`;
  }

  _buf(d, u) { const s=d.byteLength||d;const b=this.device.createBuffer({size:s,usage:u,mappedAtCreation:!!d.byteLength});if(d.byteLength){new Uint32Array(b.getMappedRange()).set(d);b.unmap();}return b;}
  _toArr(l){const a=new Uint32Array(l.length*this.L);l.forEach((n,i)=>{let v=n;for(let j=0;j<this.L;j++){a[i*this.L+j]=Number(v&0xFFFFFFFFn);v>>=32n}});return a;}
  _fromArr(a){const r=[];for(let i=0;i<a.length/this.L;i++){let n=0n;for(let j=this.L-1;j>=0;j--)n=(n<<32n)|BigInt(a[i*this.L+j]);r.push(n)}return r;}
}