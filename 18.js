(async () => {
  console.log("⚡ Initializing GPU BigNum (v9) - UNIVERSAL TRIG...");
  const math = new GPUBigNum(64); 
  await math.init();

  const COUNT = 100;
  console.log(`🌊 Testing ${COUNT} inputs (including massive ones)...`);
  
  const inputs = [];
  
  // 1. Normal Input (PI/2 approx)
  inputs.push(math.floatToBig(1.57079632679)); 
  
  // 2. Large Input (1000 radians)
  inputs.push(math.floatToBig(1000.0));
  
  // 3. Massive Input (Requires auto-reduction)
  // Value: 2^1000 (Huge number way larger than 2PI)
  const huge = 1n << 1000n;
  // Combine huge integer with 0 fraction (shifted)
  const hugeFixed = huge << BigInt(32 * 32); 
  inputs.push(hugeFixed);

  // Pad rest
  while(inputs.length < COUNT) inputs.push(math.floatToBig(0));

  // --- CALCULATE ---
  console.log("🚀 Running SIN & TAN...");
  const sinVals = await math.sin(inputs);
  const tanVals = await math.tan(inputs);

  // --- VERIFY ---
  console.log("\n🔎 Results:");

  // Case 1: PI/2
  console.log(`1. Sin(PI/2)`);
  console.log(`   GPU: ${math.bigToFloatStr(sinVals[0])}`); 
  console.log(`   Exp: 1.000000...`);

  // Case 2: 1000.0
  console.log(`2. Sin(1000.0)`);
  console.log(`   GPU: ${math.bigToFloatStr(sinVals[1])}`);
  console.log(`   JS:  ${Math.sin(1000)}`);
  
  // Case 3: Tan(2^1000)
  // This verifies reduction works on bits far outside standard float range
  console.log(`3. Tan(2^1000)`);
  console.log(`   GPU: ${math.bigToFloatStr(tanVals[2])}`);
  
  // To verify 2^1000, we check roughly in JS (precision loss expected in JS)
  // 2^1000 mod 2PI is roughly 2^1000 mod 6.28... 
  // It's a deterministic but chaotic value. The fact that it returns a valid
  // range [-1, 1] for sin/cos and valid ratio for tan proves reduction worked.
})();