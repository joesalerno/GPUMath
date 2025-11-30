
export async function runSuite(engine, math, log) {
    log("=== Running Extended Test Suite ===");
    let passes = 0;
    let fails = 0;

    const assert = (condition, msg) => {
        if (condition) {
            log(`✅ ${msg}`);
            passes++;
        } else {
            log(`❌ ${msg}`);
            fails++;
        }
    };

    const assertClose = (aBig, bFloat, epsilon = 0.0001, msg) => {
        const aFloat = parseFloat(engine.bigToFloatStr(aBig));
        const diff = Math.abs(aFloat - bFloat);
        if (diff < epsilon) {
            log(`✅ ${msg} (Got ${aFloat}, Expected ${bFloat})`);
            passes++;
        } else {
            log(`❌ ${msg} (Got ${aFloat}, Expected ${bFloat}, Diff ${diff})`);
            fails++;
        }
    };

    try {
        // Test 1: Addition
        const a1 = engine.floatToBig(10.5);
        const b1 = engine.floatToBig(20.25);
        const res1 = await math.add([a1], [b1]);
        assertClose(res1[0], 30.75, 0.0001, "Addition: 10.5 + 20.25");

        // Test 2: Subtraction
        const res2 = await math.sub([a1], [b1]);
        assertClose(res2[0], -9.75, 0.0001, "Subtraction: 10.5 - 20.25");

        // Test 3: Multiplication
        const res3 = await math.mul([a1], [b1]);
        assertClose(res3[0], 212.625, 0.001, "Multiplication: 10.5 * 20.25");

        // Test 4: Division
        // 20.25 / 10.5 = 1.9285714...
        const res4 = await math.div([b1], [a1]);
        assertClose(res4[0], 20.25/10.5, 0.001, "Division: 20.25 / 10.5");

        // Test 5: Square Root
        const val5 = engine.floatToBig(2.0);
        const res5 = await math.sqrt([val5]);
        assertClose(res5[0], 1.41421356, 0.0001, "Sqrt: sqrt(2)");

        // Test 6: Trigonometry (Sin)
        const pi = engine.floatToBig(Math.PI);
        const halfPi = engine.floatToBig(Math.PI / 2);
        const res6 = await math.sin([halfPi]);
        assertClose(res6[0], 1.0, 0.001, "Sin: sin(PI/2)");

        const res7 = await math.sin([pi]);
        assertClose(res7[0], 0.0, 0.001, "Sin: sin(PI)");

        // Test 7: ModPow (if applicable, usually for integers but let's see)
        // ModPow in shader seems to be for integer modular exponentiation (RSA style), not fixed point.
        // Let's verify op_modpow in shader assumes integers.
        // It uses mul_int_op. So it treats inputs as integers.
        // Let's test with small integers.
        const base = engine.toBuffer([123n]); // 123
        const exp = engine.toBuffer([456n]); // 456
        const mod = engine.toBuffer([789n]); // 789
        // JS BigInt modpow: 123^456 % 789
        // 123^456 is huge.
        // Let's use simple values: 2^10 % 1000 = 1024 % 1000 = 24
        const base8 = engine.toBuffer([2n]);
        const exp8 = engine.toBuffer([10n]);
        const mod8 = engine.toBuffer([1000n]);

        // We need to call runOp directly or via math wrapper if exposed.
        // math wrapper might not expose modpow.
        // Let's check gpu-operations.js

    } catch (e) {
        log(`❌ Exception during tests: ${e.message}`);
        console.error(e);
        fails++;
    }

    log(`=== Suite Finished: ${passes} Passed, ${fails} Failed ===`);
}
