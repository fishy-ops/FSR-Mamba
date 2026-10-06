// Scalar rounding follows fused.py CUDA lines 463-468; compile with -Gis.
cbuffer Frame : register(b0) {
    uint w, h, wp, hp;
    uint first_frame, depth_test, bicubic, carry_raw;
    uint nearest_sample, conf_consistent, base_gate, conf_motion;
    float box_slack, conf_max, conf_m, pre_exposure;
    uint model_space, inverted_depth, display_motion, exposure_one;
    float2 motion_scale, jitter_cancel;
    float4 phase_weight;
    int4 offsets[4]; // y,x,unused,unused
    int4 windows[4];
    float4 kernels[16];
    uint debug_view, has_exposure, robustness;
    float thin_factor;
    float coverage_floor;
    float soft_osc_threshold;
    float alpha_min, padding;
    float2 signed_jitter;
    uint kpn_residual, kpn_depth_hr;
    float stabilize, stabilize_tau, stabilize_eps, stabilize_padding;
    float foliage_strength,foliage_ema,foliage_threshold,foliage_eps;
    float foliage_spatial,foliage_history_scale,foliage_alpha_floor,fallback_strength;
    float fallback_sigma,fallback_alpha;
    uint tracker_reset,foliage_padding;
};
#ifdef FSRM_FAST
#define depth_test 1
#define bicubic 1
#define carry_raw 1
#define nearest_sample 1
#define conf_consistent 1
#define base_gate 1
#define conf_motion 0
#define model_space 0
#define debug_view 0
#define mv_dilate (robustness & 1)
#define depth_dilate 0
#define thin_lock 0
#else
#define mv_dilate (robustness & 1)
#define depth_dilate (robustness & 2)
#define thin_lock (robustness & 4)
#endif
#define depth_soft (robustness & 8)
#define coverage (robustness & 16)
#define depth_soft_osc (robustness & 32)
#define history_age (robustness & 64)
#define kpn_trunk_stride ((robustness & 128) ? 2 : 1)
#define kpn_d2s_alt (robustness & 256)
#define kpn_catmull (robustness & 512)
float rh(float x) { return f16tof32(f32tof16(x)); }
float sig(float x) { return rh(1.0 / (1.0 + exp(-x))); }
int bound(int x, int n) { return clamp(x, 0, n-1); }
float stable_lerp(float a, float b, float t) { return rh(t < .5 ? a+t*(b-a) : b-(b-a)*(1-t)); }
// FSR2 removes engine pre-exposure before applying exposure; zero texture exposure means 1.
float exposure_scale(float exposure) { return exposure_one ? 1 : ((exposure == 0 ? 1 : exposure) / pre_exposure); }
