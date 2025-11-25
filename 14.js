(async () => {
  console.log("⚡ Initializing GPU BigNum (v7) with SQRT...");
  const math = new GPUBigNum(64); // 2048-bit
  await math.init();

  const COUNT = 1000;
  console.log(`💎 Generating ${COUNT} perfect squares (approx 2048-bit)...`);

  const roots = [];
  const squares = [];

  for(let i=0; i<COUNT; i++) {
    // Generate a random ~1024-bit number
    const r = BigInt("0x1" + Array(60).fill("A").join("")); // ~1000 bits
    roots.push(r);
    squares.push(r * r); // Square it -> ~2000 bits
  }

  // 1. Calculate SQRT on GPU
  console.log("🚀 Running Batch SQRT...");
  const start = performance.now();
  
  // Pass 'squares' as input. The library ignores the second argument for unary ops.
  const calculatedRoots = await math.sqrt(squares);
  
  const time = performance.now() - start;
  console.log(`   Done in ${time.toFixed(2)}ms`);
  console.log(`   Throughput: ${(COUNT/(time/1000)).toFixed(0)} sqrts/sec`);

  // 2. Verify
  console.log("\n🔎 Verification:");
  const idx = 0;
  const expected = roots[idx];
  const actual = calculatedRoots[idx];

  console.log(`   Input (Square): ${squares[idx].toString().slice(0, 20)}...`);
  console.log(`   Expected Root:  ${expected.toString().slice(0, 20)}...`);
  console.log(`   Actual Root:    ${actual.toString().slice(0, 20)}...`);

  if (expected === actual) {
    console.log("✅ SQRT MATCH!");
  } else {
    console.error("❌ SQRT FAILED");
    console.log("Diff:", expected - actual);
  }
})();