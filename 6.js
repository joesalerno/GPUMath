(async () => {
  console.log("⚡ Initializing GPU BigNum (v3)...");
  // 64 limbs = 2048-bit input. 
  // Result of multiplication will be 4096-bit.
  const math = new GPUBigNum(64); 
  await math.init();

  // Create two massive numbers
  const A = BigInt("0x" + "E".repeat(100)); // A large number
  const B = BigInt("0x" + "2".repeat(100)); // Another large number
  
  // Generate batch of 10,000
  const count = 10_000;
  const inputsA = Array(count).fill(A);
  const inputsB = Array(count).fill(B);

  // 1. Multiply
  console.log("🚀 Batch Multiplying...");
  const start = performance.now();
  const products = await math.mul(inputsA, inputsB);
  const time = performance.now() - start;
  
  console.log(`   Calculated ${count} products (4096-bit results) in ${time.toFixed(2)}ms`);
  console.log(`   Throughput: ${(count / (time/1000)).toFixed(0)} ops/sec`);

  // 2. Verify
  const expected = A * B;
  const actual = products[0];
  
  console.log("\n🔎 Verification:");
  console.log(`   Expected: ${expected.toString().slice(0, 20)}...`);
  console.log(`   Actual:   ${actual.toString().slice(0, 20)}...`);
  
  if (actual === expected) {
    console.log("✅ SUCCESS: GPU multiplication matches CPU BigInt.");
  } else {
    console.error("❌ FAILURE: Results mismatch.");
  }
})();