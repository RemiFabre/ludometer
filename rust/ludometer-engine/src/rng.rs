//! Random number generation, in two flavours.
//!
//! * [`Mt19937`] reproduces CPython's `random.Random` bit for bit: `seed(int)`,
//!   `getrandbits`, `_randbelow_with_getrandbits`, `shuffle`, `random()`. It
//!   exists so the rules and the search can be tested *exactly* against the
//!   Python engine (a game from the same seed deals the same tiles, a search
//!   with the same determinization seed samples the same refill).
//! * [`SplitMix64`] is the production generator: one `u64` of state, a few
//!   nanoseconds per draw.
//!
//! A [`Rng`] is what a game state carries: the kind, the seed and how many
//! 32-bit outputs have been consumed so far. The Mersenne Twister's 2.5 KB of
//! state is never stored: it is rebuilt from `(seed, consumed)` when a refill
//! needs it, which happens once every ~20 real moves and never inside the
//! search (a determinization reseeds first). That keeps a state `Copy` and
//! small, which is what the tree needs.

/// CPython's `random.Random`: MT19937 with CPython's seeding and derived draws.
#[derive(Clone)]
pub struct Mt19937 {
    mt: [u32; 624],
    idx: usize,
}

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

impl Mt19937 {
    fn init_genrand(s: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = s;
        for i in 1..N {
            mt[i] = 1_812_433_253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { mt, idx: N }
    }

    fn init_by_array(key: &[u32]) -> Self {
        let mut g = Self::init_genrand(19_650_218);
        let mt = &mut g.mt;
        let mut i = 1usize;
        let mut j = 0usize;
        let klen = key.len().max(1);
        let mut k = N.max(klen);
        while k > 0 {
            let kj = if key.is_empty() { 0 } else { key[j] };
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525)))
                .wrapping_add(kj)
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= klen {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        mt[0] = 0x8000_0000;
        g.idx = N;
        g
    }

    /// `random.Random(seed)` / `random.seed(seed)` for a non-negative int.
    pub fn seed_int(seed: u64) -> Self {
        // CPython splits abs(seed) into 32-bit words, least significant first;
        // zero is the one-word key [0].
        let mut key = Vec::with_capacity(2);
        if seed == 0 {
            key.push(0);
        } else {
            let mut s = seed;
            while s != 0 {
                key.push((s & 0xffff_ffff) as u32);
                s >>= 32;
            }
        }
        Self::init_by_array(&key)
    }

    fn twist(&mut self) {
        let mt = &mut self.mt;
        for kk in 0..N - M {
            let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
        }
        for kk in N - M..N - 1 {
            let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M - N] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
        }
        let y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ if y & 1 == 1 { MATRIX_A } else { 0 };
        self.idx = 0;
    }

    /// One 32-bit output (`genrand_uint32`).
    #[inline]
    pub fn genrand_u32(&mut self) -> u32 {
        if self.idx >= N {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Throw away `n` outputs (rebuilding a stream position).
    pub fn skip(&mut self, n: u32) {
        for _ in 0..n {
            self.genrand_u32();
        }
    }

    /// `getrandbits(k)` for `1 <= k <= 64`.
    #[inline]
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        debug_assert!(k >= 1 && k <= 64);
        if k <= 32 {
            return (self.genrand_u32() >> (32 - k)) as u64;
        }
        // Little-endian words: the low word first, the last word truncated.
        let lo = self.genrand_u32() as u64;
        let rest = k - 32;
        let hi = (self.genrand_u32() >> (32 - rest)) as u64;
        lo | (hi << 32)
    }

    /// `_randbelow_with_getrandbits(n)`: uniform in `0..n` by rejection.
    #[inline]
    pub fn randbelow(&mut self, n: u64) -> u64 {
        debug_assert!(n >= 1);
        let k = bit_length(n);
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }

    /// `random.shuffle(x)`: Fisher–Yates from the top, `j = randbelow(i + 1)`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        let n = x.len();
        for i in (1..n).rev() {
            let j = self.randbelow(i as u64 + 1) as usize;
            x.swap(i, j);
        }
    }

    /// `random.random()`: 53 bits from two outputs.
    #[inline]
    pub fn random(&mut self) -> f64 {
        let a = (self.genrand_u32() >> 5) as f64;
        let b = (self.genrand_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// `int.bit_length()`.
#[inline]
fn bit_length(n: u64) -> u32 {
    64 - n.leading_zeros()
}

/// A counting wrapper: how many 32-bit outputs a sequence of draws consumed.
/// CPython's `randbelow` makes the count data-dependent (rejection), so the
/// only way to know it is to count while drawing.
pub struct CountingMt {
    pub mt: Mt19937,
    pub consumed: u32,
}

impl CountingMt {
    pub fn from(seed: u64, consumed: u32) -> Self {
        let mut mt = Mt19937::seed_int(seed);
        mt.skip(consumed);
        CountingMt { mt, consumed }
    }
    #[inline]
    fn u32(&mut self) -> u32 {
        self.consumed += 1;
        self.mt.genrand_u32()
    }
    #[inline]
    fn getrandbits(&mut self, k: u32) -> u64 {
        if k <= 32 {
            return (self.u32() >> (32 - k)) as u64;
        }
        let lo = self.u32() as u64;
        let hi = (self.u32() >> (64 - k)) as u64;
        lo | (hi << 32)
    }
    #[inline]
    pub fn randbelow(&mut self, n: u64) -> u64 {
        let k = bit_length(n);
        loop {
            let r = self.getrandbits(k);
            if r < n {
                return r;
            }
        }
    }
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.randbelow(i as u64 + 1) as usize;
            x.swap(i, j);
        }
    }
    pub fn random(&mut self) -> f64 {
        let a = (self.u32() >> 5) as f64;
        let b = (self.u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }
}

/// SplitMix64: the production generator (one `u64` of state).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        // Mix the seed once so that seeds 0, 1, 2 do not share a prefix.
        let mut s = SplitMix64(seed ^ 0x9E37_79B9_7F4A_7C15);
        s.next_u64();
        s
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `0..n` (Lemire's nearly-divisionless method).
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n >= 1);
        let mut m = (self.next_u64() as u128) * (n as u128);
        let mut l = m as u64;
        if l < n {
            let t = n.wrapping_neg() % n;
            while l < t {
                m = (self.next_u64() as u128) * (n as u128);
                l = m as u64;
            }
        }
        (m >> 64) as u64
    }
    /// Uniform in `[0, 1)` with 53 bits.
    #[inline]
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            x.swap(i, j);
        }
    }
}

/// Which generator a game state / a search uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RngKind {
    /// SplitMix64: fast, the default.
    Fast,
    /// CPython's `random.Random`: exact parity with the Python engine.
    Python,
}

/// The generator a [`crate::azul::State`] carries: tiny, `Copy`, rebuilt on use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rng {
    pub kind: RngKind,
    pub seed: u64,
    /// Python kind: 32-bit outputs consumed so far. Fast kind: the SplitMix state.
    pub consumed: u32,
    pub state: u64,
}

impl Rng {
    pub fn new(kind: RngKind, seed: u64) -> Self {
        match kind {
            RngKind::Fast => Rng { kind, seed, consumed: 0, state: SplitMix64::new(seed).0 },
            RngKind::Python => Rng { kind, seed, consumed: 0, state: 0 },
        }
    }

    /// `rng.seed(seed)`: restart the stream.
    pub fn reseed(&mut self, seed: u64) {
        *self = Rng::new(self.kind, seed);
    }

    /// `rng.shuffle(x)`.
    pub fn shuffle<T>(&mut self, x: &mut [T]) {
        match self.kind {
            RngKind::Fast => {
                let mut g = SplitMix64(self.state);
                g.shuffle(x);
                self.state = g.0;
            }
            RngKind::Python => {
                let mut g = CountingMt::from(self.seed, self.consumed);
                g.shuffle(x);
                self.consumed = g.consumed;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(seed: u64, u32s: [u32; 4], shuffle10: [usize; 10], random: f64, rr7: [u64; 5], gb40: u64, gb5: u64, first10: [usize; 10], last3: [usize; 3]) {
        let mut r = Mt19937::seed_int(seed);
        let got: Vec<u32> = (0..4).map(|_| r.genrand_u32()).collect();
        assert_eq!(got, u32s, "u32 seed {seed}");
        let mut r = Mt19937::seed_int(seed);
        let mut x: Vec<usize> = (0..10).collect();
        r.shuffle(&mut x);
        assert_eq!(x, shuffle10, "shuffle seed {seed}");
        assert_eq!(r.random(), random, "random seed {seed}");
        let got: Vec<u64> = (0..5).map(|_| r.randbelow(7)).collect();
        assert_eq!(got, rr7, "randrange seed {seed}");
        assert_eq!(r.getrandbits(40), gb40);
        assert_eq!(r.getrandbits(5), gb5);
        let mut r = Mt19937::seed_int(seed);
        let mut x: Vec<usize> = (0..100).collect();
        r.shuffle(&mut x);
        assert_eq!(&x[..10], &first10);
        assert_eq!(&x[97..], &last3);
    }

    #[test]
    fn matches_cpython_seed_0() {
        check(0, [3626764237, 1654615998, 3255389356, 3823568514], [7, 8, 1, 5, 3, 4, 2, 0, 9, 6], 0.3580493746949883, [1, 4, 1, 2, 1], 106325369465, 19, [23, 8, 11, 7, 48, 13, 1, 91, 94, 54], [53, 97, 49]);
    }
    #[test]
    fn matches_cpython_seed_12345() {
        check(12345, [1789368711, 3146859322, 43676229, 3522623596], [8, 7, 3, 5, 1, 2, 9, 4, 0, 6], 0.1616878239293682, [0, 6, 3, 2, 4], 191672427790, 19, [63, 19, 83, 87, 88, 28, 8, 76, 46, 59], [1, 93, 53]);
    }
    #[test]
    fn matches_cpython_multiword_seed() {
        check(1099511627779, [943978446, 261273136, 2359950418, 3580981848], [8, 5, 6, 7, 9, 2, 4, 1, 0, 3], 0.25922585779041674, [4, 2, 2, 6, 3], 924860693027, 17, [84, 36, 40, 66, 45, 14, 55, 94, 78, 37], [70, 7, 28]);
    }
    #[test]
    fn matches_cpython_seed_2p31m1() {
        check(2147483647, [1364760256, 4023463762, 3510513048, 516955790], [8, 7, 4, 0, 6, 9, 2, 3, 1, 5], 0.1537527905261088, [4, 3, 2, 5, 5], 672653102485, 26, [90, 86, 88, 64, 17, 12, 74, 35, 6, 54], [25, 15, 40]);
    }

    #[test]
    fn counting_mt_resumes_a_stream_exactly() {
        let mut a = CountingMt::from(7, 0);
        let mut x: Vec<u8> = (0..100).collect();
        a.shuffle(&mut x);
        let mut y = x.clone();
        // Resume from (seed, consumed) and draw again: must equal continuing.
        let mut b = CountingMt::from(7, a.consumed);
        a.shuffle(&mut x);
        b.shuffle(&mut y);
        assert_eq!(x, y);
        assert_eq!(a.consumed, b.consumed);
    }

    #[test]
    fn rng_python_kind_shuffles_like_cpython() {
        let mut r = Rng::new(RngKind::Python, 12345);
        let mut x: Vec<usize> = (0..10).collect();
        r.shuffle(&mut x);
        assert_eq!(x, [8, 7, 3, 5, 1, 2, 9, 4, 0, 6]);
    }

    #[test]
    fn splitmix_below_is_in_range_and_varies() {
        let mut g = SplitMix64::new(1);
        let mut seen = [false; 7];
        for _ in 0..1000 {
            let v = g.below(7) as usize;
            assert!(v < 7);
            seen[v] = true;
        }
        assert!(seen.iter().all(|&s| s));
        let f = g.random();
        assert!((0.0..1.0).contains(&f));
    }
}
