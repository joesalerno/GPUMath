import { expect, test, describe } from 'vitest';
import { toBuffer, fromBuffer, floatToBig, bigToFloatStr, hexToFixed } from '../src/math-utils.js';

describe('Math Utils Extra Tests', () => {
    // L must be > F to represent numbers >= 1.0
    const L = 16;
    const F = 8;

    test('toBuffer and fromBuffer round trip', () => {
        const input = [123456789n, -987654321n, 0n, -1n];
        const buffer = toBuffer(input, L);
        const output = fromBuffer(buffer, L);

        expect(output).toEqual(input);
    });

    test('floatToBig and bigToFloatStr round trip', () => {
        // Note: Precision loss is expected for floats
        const values = [1.0, -1.0, 0.5, 100.123, -0.001];

        values.forEach(v => {
            const big = floatToBig(v, F);
            const str = bigToFloatStr(big, F);
            const parsed = parseFloat(str);
            expect(parsed).toBeCloseTo(v, 4);
        });
    });

    test('hexToFixed converts correctly', () => {
        const hex = "1";
        const buffer = hexToFixed(hex, L, F);
        const val = fromBuffer(buffer, L);

        // Expected: 1n << (F*32)
        const expected = 1n << BigInt(F * 32);
        expect(val[0]).toBe(expected);
    });

    test('bigToFloatStr formatting', () => {
        const bigOne = 1n << BigInt(F * 32);
        // Expect "1.0" or "1" (implementation details)
        // Implementation replaces trailing zeros, so "1.0000" -> "1." -> "1.0" handled?
        // Code: `${sign}${intPart}.${fracStr || '0'}`
        // If fracStr is empty, returns "1.0".
        expect(bigToFloatStr(bigOne, F)).toBe("1.0");

        const bigHalf = 1n << BigInt(F * 32 - 1);
        // "0.5"
        expect(bigToFloatStr(bigHalf, F)).toBe("0.5");
    });
});
