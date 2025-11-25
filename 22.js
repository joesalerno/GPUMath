(async () => {
  console.log("🌟 Initializing Ultimate GPU Math Engine...");
  const math = new GPUMath();
  await math.init();

  // --- 1. Basic Arithmetic (Fixed Point) ---
  console.log("\n🧪 1. Arithmetic Test (A + B, A * B)");
  const A = math.floatToBig(123.456);
  const B = math.floatToBig(789.123);
  const sum = (await math.add([A], [B]))[0];
  const mul = (await math.mul([A], [B]))[0];
  console.log(`   ${math.bigToFloatStr(A)} + ${math.bigToFloatStr(B)} = ${math.bigToFloatStr(sum)}`);
  console.log(`   ${math.bigToFloatStr(A)} * ${math.bigToFloatStr(B)} = ${math.bigToFloatStr(mul)}`);

  // --- 2. Trigonometry (Scientific) ---
  console.log("\n🧪 2. Trigonometry (Sin, Cos, Tan)");
  // Test PI (approx)
  const inputs = [math.floatToBig(3.14159265), math.floatToBig(0.785398)]; // PI, PI/4
  const sines = await math.sin(inputs);
  const tans = await math.tan(inputs);
  console.log(`   sin(PI)   = ${math.bigToFloatStr(sines[0])} (Exp: ~0.0)`);
  console.log(`   tan(PI/4) = ${math.bigToFloatStr(tans[1])}  (Exp: ~1.0)`);

  // --- 3. RSA Style Crypto Math (Integer) ---
  console.log("\n🧪 3. Modular Exponentiation (RSA)");
  // C = M^E mod N
  const msg = 123456789n;
  const exp = 65537n;
  const mod = BigInt("0x" + "F".repeat(64)); // 256-bit modulus
  const encrypted = (await math.modPow([msg], [exp], [mod]))[0];
  
  // CPU Verify
  const cpuModPow = (b, e, m) => { let r=1n; b=b%m; while(e>0n){if(e%2n)r=(r*b)%m; b=(b*b)%m; e/=2n;} return r; };
  const expected = cpuModPow(msg, exp, mod);
  
  console.log(`   GPU: ${encrypted.toString().slice(0, 20)}...`);
  console.log(`   CPU: ${expected.toString().slice(0, 20)}...`);
  console.log(encrypted === expected ? "   ✅ Match" : "   ❌ Mismatch");

  // --- 4. Batch Performance ---
  console.log("\n🧪 4. Performance Stress Test");
  const COUNT = 5000;
  const hugeInputs = Array(COUNT).fill(math.floatToBig(10000.5));
  console.log(`   Calculating Sqrt for ${COUNT} huge numbers...`);
  const t0 = performance.now();
  await math.sqrt(hugeInputs);
  const t1 = performance.now();
  console.log(`   Done in ${(t1-t0).toFixed(2)}ms. Throughput: ${(COUNT/((t1-t0)/1000)).toFixed(0)} ops/sec`);

})();