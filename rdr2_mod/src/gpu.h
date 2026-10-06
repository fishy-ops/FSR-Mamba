#pragma once
#include "cpu_math.h"
#include "runtime_config.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace fsrmamba {
using Microsoft::WRL::ComPtr;
void checked(HRESULT hr, const char* operation, ID3D12Device* device=nullptr);
void drain_debug_messages(ID3D12Device* device) noexcept;
ComPtr<ID3D12Resource> buffer(ID3D12Device* device, uint64_t bytes, D3D12_HEAP_TYPE heap);
void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
void uav_barrier(ID3D12GraphicsCommandList* list);
struct alignas(16) Constants {
    uint32_t w,h,wp,hp;
    uint32_t first_frame,depth_test,bicubic,carry_raw;
    uint32_t nearest_sample,conf_consistent,base_gate,conf_motion;
    float box_slack,conf_max,conf_m,pre_exposure;
    uint32_t model_space,inverted_depth,display_motion,exposure_one;
    float motion_scale[2],jitter_cancel[2];
    float phase[4];
    int32_t offsets[4][4],windows[4][4];
    float kernels[64];
    uint32_t debug_view,has_exposure,robustness;
    float thin_factor;
    float coverage_floor;
    float soft_osc_threshold;
    float alpha_min, padding;
    float signed_jitter[2];
    uint32_t kpn_residual, kpn_depth_hr;
    float stabilize, stabilize_tau, stabilize_eps, stabilize_padding;
    float foliage_strength,foliage_ema,foliage_threshold,foliage_eps;
    float foliage_spatial,foliage_history_scale,foliage_alpha_floor,fallback_strength;
    float fallback_sigma,fallback_alpha;
    uint32_t tracker_reset,foliage_padding;
};
static_assert(sizeof(Constants)==608);
constexpr size_t constant_buffer_bytes=(sizeof(Constants)+255)&~size_t(255);
struct InputResource {
    ID3D12Resource* resource=nullptr;
    DXGI_FORMAT format=DXGI_FORMAT_UNKNOWN;
    D3D12_RESOURCE_STATES state=D3D12_RESOURCE_STATE_COMMON;
};
struct FrameInputs {
    InputResource color,motion,depth,exposure,output;
    double jitter_x=0,jitter_y=0;
    float motion_scale[2]={1,1},jitter_cancel[2]={0,0};
    float pre_exposure=1,sharpness=0;
    float stabilize=0,stabilize_tau=.5f,stabilize_eps=.004f;
    FoliageControls foliage;
    bool auto_exposure=false,sharpen=false;
    bool reset=false,model_space=true,inverted_depth=true,display_motion=false,exposure_one=true;
    uint32_t debug_view=0;
};
// Logging hook and DirectML location (set by the host before the first Pipeline is created).
void set_log_sink(void (*sink)(const char*));
void set_directml_folder(const std::filesystem::path& folder);
struct PipelineOptions {
    uint32_t ring_frames=8;
    bool depth_to_space_alt=false; // swap DML depth-to-space channel order (first-run diagnosis)
    bool fast_path=true, dml_graph=false; // opt in until validated on Windows
    bool debug_layer=false; // host enabled the D3D12 debug layer before device creation
};
class Pipeline {
public:
    Pipeline(ID3D12Device* device, const Weights& weights, uint32_t width, uint32_t height, const PipelineOptions& options={});
    ~Pipeline();
    Pipeline(const Pipeline&)=delete;
    Pipeline& operator=(const Pipeline&)=delete;
    void record(ID3D12GraphicsCommandList* list,const FrameInputs& inputs,ID3D12QueryHeap* timestamps=nullptr);
    void reset();
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
