(async () => {
  console.log("⚡ Initializing GPU BigNum (v8) with TRIGONOMETRY...");
  // 64 limbs total. 32 limbs (1024 bits) are INTEGER, 32 limbs (1024 bits) are FRACTION.
  const math = new GPUBigNum(64); 
  await math.init();

  const COUNT = 1000;
  const inputs = [];
  
  console.log(`🌊 Generating ${COUNT} angles (radians)...`);
  // Generate inputs: 0.0, 0.01, 0.02 ...
  for(let i=0; i<COUNT; i++) {
    const val = (i / 100) % 3.14159; // Simple range
    inputs.push(math.floatToBig(val));
  }

  // 1. Calculate SIN
  console.log("🚀 Running Batch SIN...");
  const start = performance.now();
  const results = await math.sin(inputs);
  const time = performance.now() - start;

  // 2. Verify Result [0]
  const val0 = 0.0; // sin(0) = 0
  const res0 = results[0];
  
  // 3. Verify Result [100] (1.0 radians)
  const val1 = 1.0; 
  const res1 = results[100];
  const expected1 = Math.sin(1.0);

  console.log(`   Done in ${time.toFixed(2)}ms`);
  
  console.log("\n🔎 Verification:");
  console.log(`   sin(0.0) -> GPU: ${math.bigToFloatStr(res0)}`);
  
  console.log(`   sin(1.0) -> GPU: ${math.bigToFloatStr(res1)}`);
  console.log(`   sin(1.0) -> JS:  ${expected1}`);

  // Note: High precision matching requires careful string parsing
  // JS float has only ~15 digits precision. GPU result has hundreds.
})();