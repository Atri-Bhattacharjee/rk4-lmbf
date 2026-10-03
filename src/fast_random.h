#pragma once

/**
 * @file fast_random.h
 * @brief Keyed (counter-style) random streams and a fast standard-normal sampler.
 *
 * Used by the opt-in fast mode of SMC_LMB_Tracker (set_fast_mode). Two properties matter there:
 *
 *  - **Keyed streams.** A stream is a pure function of a 64-bit key, so the noise a particle receives
 *    depends only on *what* is being drawn (seed, track, substep, particle), never on how many draws
 *    anything else made before it.
 *  - **Speed.** The legacy path draws through std::normal_distribution over std::mt19937_64 (the
 *    Marsaglia polar method: two uniforms, a log, a sqrt and a division per pair, with 21% rejection).
 *    That was measured at ~76 of the ~147 ns of one particle propagation step. The ziggurat below
 *    returns on its first 64-bit draw ~98.8% of the time with one multiply and one compare.
 *
 * Generator: SplitMix64 (Steele, Lea & Flood 2014; the seeding generator of xoshiro), a Weyl sequence
 * passed through a strong 64-bit finaliser. It passes BigCrush, needs 8 bytes of state, and a fresh
 * stream costs one hash, which is what makes per-particle streams affordable.
 *
 * Normal sampler: the 256-layer ziggurat of Marsaglia & Tsang (2000) with Doornik's (2005)
 * correction -- the layer index and the uniform come from disjoint bits of one draw, so they are
 * independent -- and Marsaglia's exact method for the tail beyond R. Every logarithm is taken of a
 * uniform on the OPEN interval (0, 1), so no draw can produce log(0) = -inf or a NaN.
 */

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace fast_random {

//! The SplitMix64 increment (2^64 / golden ratio).
constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

//! SplitMix64 finaliser: a bijection on 64-bit words with full avalanche.
inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

//! Fold one more word into a key. Order-sensitive, and never maps (0, 0) to a fixed point.
inline uint64_t combine(uint64_t key, uint64_t value) {
    return mix64(key ^ mix64(value + kGolden));
}

//! Bit pattern of a double, so times and states can be folded into a key. -0.0 and +0.0 are made
//! equal (they are the same time); NaNs fold as whatever bits they carry, deterministically.
inline uint64_t bits_of(double value) {
    if (value == 0.0) {
        value = 0.0;
    }
    uint64_t out = 0;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

/**
 * @brief A SplitMix64 stream. Construct from a key; every key gives an independent-looking stream.
 */
class Stream {
public:
    explicit Stream(uint64_t key) : state_(mix64(key ^ 0x6A09E667F3BCC909ULL)) {}

    uint64_t next() {
        state_ += kGolden;
        return mix64(state_);
    }

    //! Uniform on [0, 1) with 53 random bits.
    double uniform() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

    //! Uniform on the open interval (0, 1): never 0, never 1, so its log is always finite.
    double uniform_open() { return (static_cast<double>(next() >> 11) + 0.5) * 0x1.0p-53; }

    double normal();

private:
    uint64_t state_;
};

namespace detail {

constexpr int kLayers = 256;
//! Start of the tail (rightmost layer edge) and the common area of every layer, for 256 layers.
constexpr double kR = 3.6541528853610088;
constexpr double kV = 0.00492867323399;

struct ZigguratTables {
    std::array<double, kLayers + 1> x{};   //!< Layer edges; x[0] = V / f(R) is the base strip's width
    std::array<double, kLayers + 1> f{};   //!< exp(-x^2 / 2) at each edge
};

inline ZigguratTables build_tables() {
    ZigguratTables t;
    auto pdf = [](double x) { return std::exp(-0.5 * x * x); };
    t.x[0] = kV / pdf(kR);
    t.x[1] = kR;
    for (int i = 2; i < kLayers; ++i) {
        t.x[i] = std::sqrt(-2.0 * std::log(kV / t.x[i - 1] + pdf(t.x[i - 1])));
    }
    t.x[kLayers] = 0.0;
    for (int i = 0; i <= kLayers; ++i) {
        t.f[i] = pdf(t.x[i]);
    }
    return t;
}

inline const ZigguratTables& tables() {
    static const ZigguratTables t = build_tables();   // thread-safe initialisation (C++11)
    return t;
}

}  // namespace detail

inline double Stream::normal() {
    const detail::ZigguratTables& t = detail::tables();
    for (;;) {
        const uint64_t bits = next();
        const int layer = static_cast<int>(bits & 0xFF);
        // Symmetric uniform on [-1, 1) from the top 53 bits, disjoint from the layer bits.
        const double u = 2.0 * (static_cast<double>(bits >> 11) * 0x1.0p-53) - 1.0;
        const double x = u * t.x[layer];
        if (std::abs(x) < t.x[layer + 1]) {
            return x;   // inside the layer's rectangle: ~98.8% of draws
        }
        if (layer == 0) {
            // Tail beyond R (Marsaglia 1964): x = -ln(U1)/R, accept when -2 ln(U2) >= x^2.
            double tail = 0.0;
            double log_u2 = 0.0;
            do {
                tail = -std::log(uniform_open()) / detail::kR;
                log_u2 = std::log(uniform_open());
            } while (-2.0 * log_u2 < tail * tail);
            return u < 0.0 ? -(detail::kR + tail) : detail::kR + tail;
        }
        // Wedge between the rectangle and the curve.
        if (t.f[layer + 1] + (t.f[layer] - t.f[layer + 1]) * uniform() < std::exp(-0.5 * x * x)) {
            return x;
        }
    }
}

}  // namespace fast_random
