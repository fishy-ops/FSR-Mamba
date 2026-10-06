#pragma once
#include "weights.h"
#include "runtime_config.h"
#include <array>

namespace fsrmamba {
struct FrameMath {
    float thin_factor = 1, coverage_floor = 0, soft_osc_threshold = .5f, alpha_min = .05f;
    std::array<float, 2> signed_jitter;
    std::array<float, 4> phase;
    std::array<int32_t, 8> offsets, windows; // y,x per phase.
    std::array<float, 64> kernels;
    std::vector<uint16_t> head_weight, head_bias;
};
int round_even(double value);
// Persistent per-pipeline object: all buffers are sized in the constructor, compute() never allocates.
class FrameMaths {
public:
    explicit FrameMaths(const Weights& weights);
    const FrameMath& compute(double jitter_x, double jitter_y);
private:
    const Weights* w;
    std::vector<float> out_w, out_b, film_a, film_b, film_c, film_d, gb, weight, bias;
    float sharp;
    FrameMath result{};
};
FrameMath frame_math(const Weights& weights, double jitter_x, double jitter_y);
float exposure_scale(float pre_exposure, float exposure, bool use_fsr);
struct KPNPixel {
    std::array<float, 3> color;
    float alpha;
};
// Params are seven decoded head values; history is one fp32 RGB pixel, y/x are output coordinates.
KPNPixel kpn_pixel(const Config& config, const float* rgb, const float* history, const float* params,
                   std::array<float, 2> signed_jitter, int h, int w, int y, int x, bool reset,
                   const FoliageControls& controls={}, std::array<float,3> evidence={});
struct FoliageTracker { std::array<float,4> state; float confidence; };
FoliageTracker foliage_tracker(float luma,float lowpass,float range,const std::array<float,4>& old,
                                float old_lowpass,float motion_spread,bool invalid,const FoliageControls& controls);
struct RobustPixel {
    std::array<float, 2> motion;
    float depth, thin;
    std::array<float, 3> slack;
};
// Model-space RGB/depth are rounded to half, motion remains float32. RGB is HWC.
RobustPixel robust_pixel(const Weights& weights, const float* rgb, const float* motion,
                         const float* depth, int h, int w, int y, int x);
struct CoveragePixel {
    std::array<float, 2> evidence;
    std::array<float, 4> state; // m1, m2, oscillation, instantaneous luma (float32).
};
CoveragePixel coverage_pixel(const float* rgb, const float* previous, std::array<float, 2> motion,
                             int h, int w, int y, int x, bool reset);
struct AgePixel { float age, next, feature; };
AgePixel age_pixel(const float* previous, std::array<float, 2> motion,
                   int h, int w, int y, int x, bool reset);
float age_alpha(float alpha, float age, float alpha_min, bool reset);
float coverage_floor(float bias);
float coverage_coefficient(float logit, float bias=-6);
// Hard deployment gate. Depth bounds/products are half rounded, previous depth is float32.
bool depth_reset(const Weights& weights, bool offscreen, bool first, float previous_depth,
                 float depth_min, float depth_max, float osc_n);
}
