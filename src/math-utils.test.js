import { describe, it, expect } from 'vitest';
import { toBuffer, fromBuffer, floatToBig, bigToFloatStr } from './math-utils.js';

describe('math-utils', () => {
    describe('toBuffer and fromBuffer', () => {
        it('should correctly convert BigInts to buffer and back', () => {
            const L = 4; // 4 limbs = 128 bits
            const values = [0n, 1n, 123456789n, -1n, -123n];

            // Adjust values for fixed size limbs manually for expectation or trust round trip
            // toBuffer handles negative by standard 2's complement if properly masked
            // But javascript BigInts are arbitrary precision.
            // toBuffer implementation:
            /*
            bigInts.forEach((bn, i) => {
                let n = bn;
                for (let j = 0; j < L; j++) {
                    arr[i * L + j] = Number(n & 0xFFFFFFFFn);
                    n >>= 32n;
                }
            });
            */
            // This logic correctly slices the BigInt into 32-bit chunks.
            // Negative numbers in BigInt: -1n is ...11111111.
            // (-1n & 0xFFFFFFFFn) is 0xFFFFFFFFn.
            // (-1n >> 32n) is -1n.
            // So it works for 2's complement representation.

            const buffer = toBuffer(values, L);
            expect(buffer).toBeInstanceOf(Uint32Array);
            expect(buffer.length).toBe(values.length * L);

            const result = fromBuffer(buffer, L);

            // fromBuffer logic checks MSB of last limb for sign extension.
            // L=4, last limb index is 3.

            expect(result[0]).toBe(0n);
            expect(result[1]).toBe(1n);
            expect(result[2]).toBe(123456789n);

            // For negative numbers, we need to ensure the BigInt matches the expected 128-bit signed integer value
            // Since JS BigInts are arbitrary, -1n is simply -1n.
            // The round trip should preserve the value if it fits in L*32 bits.
            expect(result[3]).toBe(-1n);
            expect(result[4]).toBe(-123n);
        });

        it('should handle large positive numbers', () => {
            const L = 4;
            // Max positive 127-bit number
            const val = (1n << 126n) - 1n;
            const buffer = toBuffer([val], L);
            const res = fromBuffer(buffer, L);
            expect(res[0]).toBe(val);
        });

        it('should handle large negative numbers', () => {
            const L = 4;
            // Min negative 128-bit number: -2^127
            const val = -(1n << 127n);
            const buffer = toBuffer([val], L);
            const res = fromBuffer(buffer, L);
            expect(res[0]).toBe(val);
        });
    });

    describe('floatToBig', () => {
        it('should convert float to fixed point BigInt', () => {
            const F = 2; // 64 fractional bits
            const val = 1.5;
            // 1.5 * 2^(2*32) = 1.5 * 2^64
            // 2^64 = 18446744073709551616
            // 1.5 * 2^64 = 27670116110564327424

            const big = floatToBig(val, F);
            const expected = BigInt(Math.round(val * Number(1n << 52n))) << (BigInt(F * 32) - 52n);

            expect(big).toBe(expected);
        });

        it('should handle negative floats', () => {
            const F = 2;
            const val = -1.25;
            const big = floatToBig(val, F);
            const expected = BigInt(Math.round(val * Number(1n << 52n))) << (BigInt(F * 32) - 52n);
            expect(big).toBe(expected);
        });
    });

    describe('bigToFloatStr', () => {
        it('should convert fixed point BigInt to string', () => {
            const F = 1; // 32 fractional bits
            const val = 1.5;
            const big = floatToBig(val, F);
            const str = bigToFloatStr(big, F);
            expect(str).toBe("1.5");
        });

        it('should convert negative fixed point BigInt to string', () => {
            const F = 1;
            const val = -123.456;
            const big = floatToBig(val, F);
            const str = bigToFloatStr(big, F);
            // Precision might vary slightly due to float representation
            expect(str).toMatch(/^-123\.456/);
        });

        it('should handle small numbers', () => {
             const F = 2; // Increase precision
             const val = 0.001;
             const big = floatToBig(val, F);
             const str = bigToFloatStr(big, F);
             // Floating point conversion might result in 0.000999... or 0.001...
             expect(str).toMatch(/^0\.00(1|0999)/);
        });
    });
});
