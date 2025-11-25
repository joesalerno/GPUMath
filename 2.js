(async () => {
  console.log("Initializing GPU...");
  const math = new GPUBigNum(64); // 64 limbs = 2048 bits
  await math.init();

  // 1. Generate huge test data
  const COUNT = 100_000;
  const inputsA = [];
  const inputsB = [];
  
  console.log(`Generating ${COUNT} pairs of 2048-bit numbers...`);
  for(let i=0; i<COUNT; i++) {
    // Create random huge numbers
    inputsA.push(BigInt("0x" + Array(500).fill("F").join(""))); 
    inputsB.push(1n); // Adding 1 to max value to trigger cascades
  }

  // 2. Run on GPU
  console.log("Running on GPU...");
  const start = performance.now();
  const results = await math.batchAdd(inputsA, inputsB);
  const end = performance.now();

  // 3. Verify
  console.log(`Done in ${(end - start).toFixed(2)}ms`);
  console.log(`Result[0] sample: ${results[0].toString().slice(0, 20)}...`);
  
  // Sanity check in JS
  const expected = inputsA[0] + inputsB[0];
  if (results[0] === expected) {
    console.log("✅ Verification SUCCESS!");
  } else {
    console.error("❌ Verification FAILED");
    console.log("Expected:", expected);
    console.log("Got:     ", results[0]);
  }
})();