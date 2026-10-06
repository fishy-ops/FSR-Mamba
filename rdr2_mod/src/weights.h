#pragma once
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace fsrmamba {
struct Config {
    std::string arch = "fast", preset;
    std::vector<int> widths, depths, render_size, output_size, padded_size;
    bool lite = false, residual = true;
    int trunk_stride = 1, taps = 5;
    std::string history_filter = "bicubic";
    bool kpn() const { return arch == "kpn"; }
    bool film = false, depth_test = false, nearest_sample = false;
    bool conf_consistent = false, carry_raw = false, base_gate = false;
    bool bicubic = false, conf_motion = false, coverage = false;
    bool depth_soft = false, mv_dilate = false, depth_dilate = false, thin_lock = false;
    bool depth_soft_osc = false, history_age = false;
    float jitter_sign = 1, conf_max = 32, coverage_bias = -6;
    float sigma_min = .3f, proximity = 0, proximity_gain = 2;
    int input_channels() const { return kpn() ? 16 * trunk_stride * trunk_stride : 4 * (25 + (carry_raw ? 4 : 0) + (thin_lock ? 1 : 0) + (coverage ? 2 : 0) + (history_age ? 1 : 0)); }
    int output_channels() const { return kpn() ? 28 * trunk_stride * trunk_stride : 4 * (20 + (carry_raw ? 4 : 0) + (base_gate ? 4 : 0) + (coverage ? 4 : 0)); }
};
struct Tensor {
    std::vector<uint32_t> shape;
    uint32_t dtype = 0; // 1: IEEE binary16, 2: IEEE binary32.
    std::vector<uint8_t> bytes;
    std::vector<float> floats() const;
};
struct Weights {
    Config config;
    std::map<std::string, Tensor> tensors;
    static Weights read(const std::filesystem::path& path);
    const Tensor& at(const std::string& name) const;
    float scalar(const std::string& name) const;
    void validate() const;
};
uint16_t to_half(float value);
float from_half(uint16_t value);
}
