#include "gpu.h"
#include "file_hash.h"
#include "dump.h"
#include "ffx_fsr2.h"
#include "ffx_assert.h"
#include "runtime_config.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <mutex>
#include <new>
#include <stdexcept>

namespace {
using namespace fsrmamba;
#define EXPORTS(X) \
 X(ffxAssertReport) X(ffxAssertSetPrintingCallback) X(ffxFsr2ContextCreate) \
 X(ffxFsr2ContextDestroy) X(ffxFsr2ContextDispatch) X(ffxFsr2ContextGenerateReactiveMask) \
 X(ffxFsr2GetJitterOffset) X(ffxFsr2GetJitterPhaseCount) \
 X(ffxFsr2GetRenderResolutionFromQualityMode) X(ffxFsr2GetUpscaleRatioFromQualityMode) X(ffxFsr2ResourceIsNull)
struct FrameSkip : std::runtime_error { using std::runtime_error::runtime_error; };
struct Retired { uint64_t until; std::unique_ptr<Pipeline> pipeline; };
struct Context {
    FfxFsr2ContextDescription desc{};
    ComPtr<ID3D12Device> device;
    std::unique_ptr<Pipeline> pipeline;
    std::vector<Retired> retired;
    uint64_t dispatches=0,generation=0;
    uint32_t width=0,height=0;
    FfxFloatCoords2D jitter{};
    bool ours=false;
    DumpQueue dumps;
};
void runtime_log(const char* m);
struct PresetSound { unsigned count; HMODULE module; };
DWORD WINAPI preset_sound(void* argument) {
    const auto sound=*static_cast<PresetSound*>(argument);
    delete static_cast<PresetSound*>(argument);
    static std::mutex mutex;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for(unsigned i=0;i<sound.count;++i) { Beep(880,80); if(i+1<sound.count) Sleep(100); }
    }
    FreeLibraryAndExitThread(sound.module,0);
    return 0;
}
void beep_preset(unsigned count) noexcept {
    HMODULE module=nullptr;
    if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
        reinterpret_cast<LPCWSTR>(&preset_sound),&module)) return;
    auto* sound=new(std::nothrow) PresetSound{count,module};
    if(sound) {
        HANDLE thread=CreateThread(nullptr,0,preset_sound,sound,0,nullptr);
        if(thread) { CloseHandle(thread); return; }
        delete sound;
    }
    FreeLibrary(module);
}
struct Runtime : LiveControls {
#define FIELD(name) decltype(&name) name=nullptr;
    EXPORTS(FIELD)
#undef FIELD
    std::recursive_mutex mutex;
    std::map<FfxFsr2Context*,Context> contexts;
    std::unique_ptr<Weights> weights;
    std::filesystem::path folder;
    bool enabled=true,failed=false,logging=true,key_down=false,exposure_one=false,skip_logged=false,d2s_alt=false;
    float jitter_flip[2]={1,1},mv_flip[2]={1,1};
    uint32_t ring_frames=8,debug_view=0,dump_frames=0,dump_every=1;
    DumpRun dump{this,[](void* owner,const char* message) noexcept {
        auto& r=*static_cast<Runtime*>(owner); r.lines=std::min(r.lines,199u); r.log(message);
    }};
    int toggle_key=VK_END,preset_key=0xdb;
    bool preset_down=false;
    PresetCycle presets;
    LiveControls ini_controls;
    std::string ini_hash;
    uint64_t generation=0;
    unsigned lines=0;
    void log(const char* message,bool error=false) noexcept {
        if(!logging || (!error && lines>=20000)) return;
        try { std::ofstream f(folder/L"fsrmamba.log",std::ios::app); f<<(error?"ERROR: ":"")<<message<<'\n'; ++lines; } catch(...) {}
    }
    void fail(const char* message) noexcept { if(!failed) log(message,true); failed=true; }
    Runtime() noexcept {
        try {
            HMODULE self=nullptr; wchar_t path[32768];
            if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                reinterpret_cast<LPCWSTR>(&ffxFsr2ContextCreate),&self)) throw std::runtime_error("proxy module unavailable");
            DWORD n=GetModuleFileNameW(self,path,32768); if(!n || n>=32768) throw std::runtime_error("proxy path unavailable");
            folder=std::filesystem::path(path).parent_path();
            auto original=folder/L"ffx_fsr2_api_x64.orig.dll";
            HMODULE module=LoadLibraryExW(original.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
            if(!module) throw std::runtime_error("cannot load ffx_fsr2_api_x64.orig.dll");
            bool complete=true;
#define LOAD(name) name=reinterpret_cast<decltype(name)>(GetProcAddress(module,#name)); complete=complete && name;
            EXPORTS(LOAD)
#undef LOAD
            if(!complete) throw std::runtime_error("original DLL is missing exports");
            auto ini=(folder/L"fsrmamba.ini").wstring();
            auto integer=[&](const wchar_t* key,int value){return GetPrivateProfileIntW(L"fsrmamba",key,value,ini.c_str());};
            enabled=integer(L"enable",1)!=0; ini_enable=enabled; logging=integer(L"log",1)!=0; debug_view=integer(L"debug_view",0);
            ring_frames=integer(L"ring_frames",8);
            read_dump_ini(ini);
            read_stabilize_ini(ini);
            wchar_t key[64]; GetPrivateProfileStringW(L"fsrmamba",L"toggle_key",L"VK_END",key,64,ini.c_str());
            toggle_key=runtime_key(key);
            if(!toggle_key) throw std::runtime_error("invalid toggle_key");
            wchar_t mode[64]; GetPrivateProfileStringW(L"fsrmamba",L"exposure_mode",L"fsr",mode,64,ini.c_str());
            if(std::wstring(mode)!=L"fsr" && std::wstring(mode)!=L"one") throw std::runtime_error("invalid exposure_mode");
            exposure_one=std::wstring(mode)==L"one";
            if(ring_frames<2 || ring_frames>64 || debug_view>2) throw std::runtime_error("invalid ring_frames or debug_view");
            const std::string hash=file_hash(original);
            const bool known=hash=="887d11aaef717aa2d4713f4b6fb56b3d4c75737af51042de22f70f0a3fb89f83";
            log(("ffx_fsr2_api_x64.orig.dll sha256 "+hash+(known?" (known FSR 2.2.1 build)":" (UNKNOWN build)")).c_str());
            if(!known && integer(L"require_known_fsr2",1)!=0) throw std::runtime_error("unverified original DLL ABI (set require_known_fsr2=0 to override); forwarding only");
            jitter_flip[0]=integer(L"jitter_flip_x",0)?-1.f:1.f; jitter_flip[1]=integer(L"jitter_flip_y",0)?-1.f:1.f;
            mv_flip[0]=integer(L"mv_flip_x",0)?-1.f:1.f; mv_flip[1]=integer(L"mv_flip_y",0)?-1.f:1.f;
            d2s_alt=integer(L"depth_to_space_alt",0)!=0; dml_graph=integer(L"dml_graph",1)!=0;
            set_log_sink([](const char* m){ runtime_log(m); }); set_directml_folder(folder);
            wchar_t name[1024]; GetPrivateProfileStringW(L"fsrmamba",L"weights",L"fsrmamba_weights.bin",name,1024,ini.c_str());
            std::filesystem::path filename(name);
            if(filename.empty() || filename.has_parent_path() || filename==L"." || filename==L"..") throw std::runtime_error("weights must name a file beside the DLL");
            weights=std::make_unique<Weights>(Weights::read(folder/filename)); weights_name=filename.wstring();
            log("FSR-Mamba loaded; verified FSR 2.2.1 ABI; END toggles learned dispatch.");
        } catch(const std::exception& e) { fail(e.what()); } catch(...) { fail("initialization failed"); }
    }
    void read_dump_ini(const std::wstring& ini) {
        const int n=static_cast<int>(GetPrivateProfileIntW(L"fsrmamba",L"dump_frames",0,ini.c_str()));
        const int k=static_cast<int>(GetPrivateProfileIntW(L"fsrmamba",L"dump_every",1,ini.c_str()));
        wchar_t path[32768];
        GetPrivateProfileStringW(L"fsrmamba",L"dump_dir",L"fsrmamba_dump",path,32768,ini.c_str());
        std::filesystem::path dir(path); if(dir.empty()) dir=L"fsrmamba_dump";
        if(dir.is_relative()) dir=folder/dir;
        const uint32_t frames=n>0?uint32_t(n):0;
        dump_every=k>0?uint32_t(k):1;
        if(dump_frames && !frames) dump.cancel();
        if(!dump_frames && frames) dump.start(frames,dump_every,dir);
        dump_frames=frames;
    }
    void apply_controls(const LiveControls& controls) {
        std::string changes;
        auto copy=controls;
        for(size_t i=0;i<live_setting_count;++i) {
            const float value=live_setting_value(copy,i);
            if(value!=live_setting_value(*this,i)) {
                live_setting_value(*this,i)=value;
                std::string key; for(const wchar_t* p=live_setting_name(i);*p;++p) key+=char(*p);
                changes+=" "+key+"="+std::to_string(value);
            }
        }
        if(!changes.empty()) log(("live controls:"+changes).c_str());
    }
    void read_stabilize_ini(const std::wstring& ini) {
        const auto hash=std::filesystem::exists(ini) ? file_hash(ini):std::string("missing");
        const bool changed=hash!=ini_hash;
        if(changed) {
            // Flush the profile cache before reading changed file contents.
            WritePrivateProfileStringW(nullptr,nullptr,nullptr,ini.c_str());
            presets.ini_reload(true);
            for(size_t i=0;i<live_setting_count;++i) {
                wchar_t text[64];
                const auto count=GetPrivateProfileStringW(L"fsrmamba",live_setting_name(i),L"",text,64,ini.c_str());
                apply_live_setting(ini_controls,i,count<63 ? text:L"");
            }
            wchar_t key[64]; GetPrivateProfileStringW(L"fsrmamba",L"preset_key",L"0xDB",key,64,ini.c_str());
            const int requested=runtime_key(key);
            preset_key=requested ? requested:0xdb;
            if(!requested) log("invalid preset_key; using 0xDB");
            for(size_t i=0;i<presets.presets.size();++i) {
                const auto name=L"preset"+std::to_wstring(i+1);
                wchar_t text[2048];
                const auto count=GetPrivateProfileStringW(L"fsrmamba",name.c_str(),L"",text,2048,ini.c_str());
                const auto result=parse_preset(count<2047 ? text:L"=",presets.presets[i]);
                if(result==PresetParse::invalid) log(("preset "+std::to_string(i+1)+" invalid; skipped").c_str());
            }
            ini_hash=hash;
        }
        apply_controls(presets.apply(ini_controls));
    }
    void poll_presets() noexcept {
        const bool down=(GetAsyncKeyState(preset_key)&0x8000)!=0;
        const bool pressed=down && !preset_down; preset_down=down;
        if(!pressed) return;
        try {
            const int n=presets.advance();
            if(!n) return;
            const auto controls=presets.apply(ini_controls);
            static_cast<LiveControls&>(*this)=controls;
            std::string message="preset "+std::to_string(n)+" applied:";
            auto copy=controls;
            for(size_t i=0;i<live_setting_count;++i) if(presets.presets[n-1].defined[i]) {
                std::string key; for(const wchar_t* p=live_setting_name(i);*p;++p) key+=char(*p);
                message+=" "+key+"="+std::to_string(live_setting_value(copy,i));
            }
            log(message.c_str());
            beep_preset(unsigned(n));
        } catch(...) { log("preset application failed",true); }
    }
    void poll() {
        bool down=(GetAsyncKeyState(toggle_key)&0x8000)!=0;
        if(down && !key_down) { enabled=!enabled; ++generation; log(enabled?"Learned upscaler enabled":"Original FSR 2 enabled"); }
        key_down=down;
        // Diagnosis aid: pick up fsrmamba.ini edits to the first-run switches without restarting the game.
        if(++poll_count%120==0) {
            auto ini=(folder/L"fsrmamba.ini").wstring();
            auto integer=[&](const wchar_t* key,int value){return GetPrivateProfileIntW(L"fsrmamba",key,value,ini.c_str());};
            read_dump_ini(ini);
            try { read_stabilize_ini(ini); }
            catch(...) { log("live controls reload failed",true); }
            const uint32_t view=uint32_t(integer(L"debug_view",0))<=2?uint32_t(integer(L"debug_view",0)):0;
            const float jf[2]={integer(L"jitter_flip_x",0)?-1.f:1.f,integer(L"jitter_flip_y",0)?-1.f:1.f};
            const float mf[2]={integer(L"mv_flip_x",0)?-1.f:1.f,integer(L"mv_flip_y",0)?-1.f:1.f};
            { // Live model swap: a new weights= file is loaded and every context rebuilds its pipeline.
                wchar_t wn[1024]; GetPrivateProfileStringW(L"fsrmamba",L"weights",L"fsrmamba_weights.bin",wn,1024,ini.c_str());
                std::wstring requested(wn);
                if(!weights_name.empty() && requested!=weights_name) {
                    weights_name=requested;
                    try {
                        std::filesystem::path f(requested);
                        if(f.empty() || f.has_parent_path()) throw std::runtime_error("weights must name a file beside the DLL");
                        weights=std::make_unique<Weights>(Weights::read(folder/f)); ++generation; lines=std::min(lines,19980u);
                        log(("weights reloaded: "+f.u8string()).c_str());
                    } catch(const std::exception& e) { log((std::string("weights reload failed: ")+e.what()).c_str()); }
                }
            }
            const bool on=integer(L"enable",1)!=0;
            { const bool rc=integer(L"rcas",0)!=0; if(rc!=rcas) { rcas=rc; log(rcas?"rcas on":"rcas off"); } }
            wchar_t gb[64]; GetPrivateProfileStringW(L"fsrmamba",L"exposure_gain",L"1",gb,64,ini.c_str()); float gain=float(_wtof(gb)); if(!(gain>0) || !std::isfinite(gain)) gain=1;
            if(view!=debug_view || jf[0]!=jitter_flip[0] || jf[1]!=jitter_flip[1] || mf[0]!=mv_flip[0] || mf[1]!=mv_flip[1] || on!=ini_enable || gain!=exposure_gain) {
                exposure_gain=gain;
                debug_view=view; jitter_flip[0]=jf[0];jitter_flip[1]=jf[1];mv_flip[0]=mf[0];mv_flip[1]=mf[1];
                if(on!=ini_enable) { ini_enable=on; enabled=on; }
                ++generation; lines=std::min(lines,19980u);
                log(("ini reloaded: enable="+std::to_string(enabled)+" debug_view="+std::to_string(debug_view)+" jitter_flip="+std::to_string(int(jf[0]))+","+std::to_string(int(jf[1]))+" mv_flip="+std::to_string(int(mf[0]))+","+std::to_string(int(mf[1]))+" exposure_gain="+std::to_string(exposure_gain)).c_str());
            }
        }
        poll_presets();
    }
    bool rcas=false, dml_graph=true; std::wstring weights_name; uint64_t poll_count=0; bool ini_enable=true; unsigned diag=0; float exposure_gain=1;
};
Runtime& runtime() { static Runtime r; return r; }

void runtime_log(const char* m) { runtime().log(m); }
std::string context_flags(uint32_t flags) {
    const char* names[]={"FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE","FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS",
        "FFX_FSR2_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION","FFX_FSR2_ENABLE_DEPTH_INVERTED",
        "FFX_FSR2_ENABLE_DEPTH_INFINITE","FFX_FSR2_ENABLE_AUTO_EXPOSURE","FFX_FSR2_ENABLE_DYNAMIC_RESOLUTION",
        "FFX_FSR2_ENABLE_TEXTURE1D_USAGE","FFX_FSR2_ENABLE_DEBUG_CHECKING"};
    std::string out;
    for(unsigned i=0;i<9;++i) if(flags&(1u<<i)) { if(!out.empty()) out+='|'; out+=names[i]; }
    if(flags&~0x1ffu) { if(!out.empty()) out+='|'; out+="UNKNOWN_BITS"; }
    return out.empty()?"NONE":out;
}
bool readable(const void* ptr,size_t bytes) {
    auto at=reinterpret_cast<uintptr_t>(ptr),end=at+bytes;
    if(!at || end<at) return false;
    while(at<end) {
        MEMORY_BASIC_INFORMATION m{};
        if(!VirtualQuery(reinterpret_cast<void*>(at),&m,sizeof(m)) || m.State!=MEM_COMMIT || (m.Protect&(PAGE_GUARD|PAGE_NOACCESS))) return false;
        auto next=reinterpret_cast<uintptr_t>(m.BaseAddress)+m.RegionSize; if(next<=at) return false; at=next;
    }
    return true;
}
bool dx12_backend(const FfxFsr2ContextDescription& d,const std::filesystem::path& folder) {
    HMODULE module=nullptr;
    if(!d.callbacks.fpCreateBackendContext || !GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(d.callbacks.fpCreateBackendContext),&module)) return false;
    auto expected=GetModuleHandleW((folder/L"ffx_fsr2_api_dx12_x64.dll").c_str());
    return expected && module==expected;
}
InputResource resource(const FfxResource& r) {
    if(!r.resource) return {};
    D3D12_RESOURCE_STATES state;
    switch(r.state) {
    case FFX_RESOURCE_STATE_GENERIC_READ: state=D3D12_RESOURCE_STATE_GENERIC_READ; break;
    case FFX_RESOURCE_STATE_UNORDERED_ACCESS: state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS; break;
    case FFX_RESOURCE_STATE_COMPUTE_READ: state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE; break;
    case FFX_RESOURCE_STATE_COPY_SRC: state=D3D12_RESOURCE_STATE_COPY_SOURCE; break;
    case FFX_RESOURCE_STATE_COPY_DEST: state=D3D12_RESOURCE_STATE_COPY_DEST; break;
    default: throw FrameSkip("unrecognized FfxResource state; using original FSR 2 for this frame");
    }
    auto* p=static_cast<ID3D12Resource*>(r.resource); auto desc=dump_resource_desc(p); auto format=desc.Format;
    if(desc.Width!=r.description.width || desc.Height!=r.description.height) throw std::runtime_error("FfxResource extent mismatch");
    switch(format) {
    case DXGI_FORMAT_R32_TYPELESS: case DXGI_FORMAT_D32_FLOAT: format=DXGI_FORMAT_R32_FLOAT; break;
    case DXGI_FORMAT_R24G8_TYPELESS: case DXGI_FORMAT_D24_UNORM_S8_UINT: format=DXGI_FORMAT_R24_UNORM_X8_TYPELESS; break;
    case DXGI_FORMAT_R32G8X24_TYPELESS: case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: format=DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS; break;
    case DXGI_FORMAT_R16_TYPELESS: case DXGI_FORMAT_D16_UNORM: format=DXGI_FORMAT_R16_UNORM; break;
    case DXGI_FORMAT_R8G8B8A8_TYPELESS: format=DXGI_FORMAT_R8G8B8A8_UNORM; break;
    case DXGI_FORMAT_R16G16B16A16_TYPELESS: format=DXGI_FORMAT_R16G16B16A16_FLOAT; break;
    default: break;
    }
    return {p,format,state};
}
void retire(Context& c,uint32_t n) {
    if(c.pipeline) { c.retired.push_back({c.dispatches+n+2,nullptr}); c.retired.back().pipeline=std::move(c.pipeline); }
}
}
extern "C" {
FfxErrorCode ffxFsr2ContextCreate(FfxFsr2Context* context,const FfxFsr2ContextDescription* d) {
    auto& r=runtime(); if(!r.ffxFsr2ContextCreate) return FFX_ERROR_BACKEND_API_ERROR;
    auto result=r.ffxFsr2ContextCreate(context,d); if(result!=FFX_OK) return result;
    std::lock_guard<std::recursive_mutex> lock(r.mutex);
    try {
        if(!readable(d,sizeof(*d))) throw std::runtime_error("unreadable context description");
        char flags[1024];
        snprintf(flags,sizeof flags,"context %p: flags 0x%x [%s]",static_cast<void*>(context),unsigned(d->flags),context_flags(d->flags).c_str());
        r.log(flags);
        if(!dx12_backend(*d,r.folder)) { r.log("Non-DX12 context: forwarding"); return result; }
        if((d->flags&~0x1ffu) || !d->device || !d->displaySize.width || !d->displaySize.height || d->displaySize.width>16384 || d->displaySize.height>16384 || !d->maxRenderSize.width || !d->maxRenderSize.height || d->maxRenderSize.width>16384 || d->maxRenderSize.height>16384) throw std::runtime_error("implausible context description");
        auto& c=r.contexts[context]; c.desc=*d; c.generation=r.generation;
        checked(static_cast<IUnknown*>(d->device)->QueryInterface(IID_PPV_ARGS(&c.device)),"Query D3D12 device");
        c.width=d->displaySize.width/2; c.height=d->displaySize.height/2;
        if(!r.failed && r.weights && !(d->displaySize.width%2) && !(d->displaySize.height%2)) c.pipeline=std::make_unique<Pipeline>(c.device.Get(),*r.weights,c.width,c.height,PipelineOptions{r.ring_frames,r.d2s_alt,true,r.dml_graph});
        r.log("DX12 context ready; learned dispatch requires exact 2x render size.");
    } catch(const std::exception& e) { r.fail(e.what()); } catch(...) { r.fail("context initialization failed"); }
    return result;
}
FfxErrorCode ffxFsr2ContextDispatch(FfxFsr2Context* context,const FfxFsr2DispatchDescription* d) {
    auto& r=runtime(); if(!r.ffxFsr2ContextDispatch) return FFX_ERROR_BACKEND_API_ERROR;
    std::lock_guard<std::recursive_mutex> lock(r.mutex);
    auto it=r.contexts.find(context); Context* c=it==r.contexts.end()?nullptr:&it->second;
    bool reset_original=c && c->ours;
    const FfxFloatCoords2D previous_jitter=c?c->jitter:FfxFloatCoords2D{};
    try {
        r.poll();
        if(c) {
            ++c->dispatches;
            c->retired.erase(std::remove_if(c->retired.begin(),c->retired.end(),[&](const Retired& old){return old.until<=c->dispatches;}),c->retired.end());
            if(c->generation!=r.generation) { retire(*c,r.ring_frames); c->generation=r.generation; }
        }
        if(c && !r.dump.forwarding() && !r.failed && r.enabled && r.weights) {
            if(!readable(d,sizeof(*d))) throw std::runtime_error("unreadable dispatch description");
            const auto w=d->renderSize.width,h=d->renderSize.height;
            if(w*uint64_t(2)==c->desc.displaySize.width && h*uint64_t(2)==c->desc.displaySize.height) {
                if(!d->commandList || w>c->desc.maxRenderSize.width || h>c->desc.maxRenderSize.height || !std::isfinite(d->preExposure) || d->preExposure<0 || !std::isfinite(d->motionVectorScale.x) || !std::isfinite(d->motionVectorScale.y)) throw std::runtime_error("implausible dispatch description");
                if(!c->pipeline || c->width!=w || c->height!=h) {
                    retire(*c,r.ring_frames); c->pipeline=std::make_unique<Pipeline>(c->device.Get(),*r.weights,w,h,PipelineOptions{r.ring_frames,r.d2s_alt,true,r.dml_graph}); c->width=w;c->height=h;
                }
                FrameInputs in; in.color=resource(d->color);in.depth=resource(d->depth);in.motion=resource(d->motionVectors);in.output=resource(d->output);in.exposure=resource(d->exposure);
                in.jitter_x=d->jitterOffset.x*r.jitter_flip[0];in.jitter_y=d->jitterOffset.y*r.jitter_flip[1];in.pre_exposure=(d->preExposure==0?1:d->preExposure)/r.exposure_gain;
                in.sharpen=r.rcas && d->enableSharpening; in.sharpness=d->sharpness; in.reset=d->reset || !c->ours;in.model_space=false;in.exposure_one=r.exposure_one;in.debug_view=r.debug_view;
                in.foliage=r.foliage;in.stabilize=r.stabilize;in.stabilize_tau=r.stabilize_tau;in.stabilize_eps=r.stabilize_eps;
                in.inverted_depth=(c->desc.flags&FFX_FSR2_ENABLE_DEPTH_INVERTED)!=0;
                in.display_motion=(c->desc.flags&FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS)!=0;
                float mx=float(in.display_motion?2*w:w),my=float(in.display_motion?2*h:h);
                in.motion_scale[0]=r.mv_flip[0]*d->motionVectorScale.x/mx;in.motion_scale[1]=r.mv_flip[1]*d->motionVectorScale.y/my;
                if(r.diag<16) { ++r.diag; char b[2048];
                    const auto reactive=d->reactive.resource?dump_resource_desc(static_cast<ID3D12Resource*>(d->reactive.resource)):D3D12_RESOURCE_DESC{};
                    const auto composition=d->transparencyAndComposition.resource?dump_resource_desc(static_cast<ID3D12Resource*>(d->transparencyAndComposition.resource)):D3D12_RESOURCE_DESC{};
                    snprintf(b,sizeof b,"frame %u: render %ux%u display %ux%u jitter %.4f %.4f mvscale %.3f %.3f preExp %.4f dt %.2f reset %d flags 0x%x near %.4f far %.4f fov %.4f sharpen %d sharpness %.4f fmt color %d depth %d mv %d out %d exposure %d reactive present=%d fmt=%u size=%llux%u transparencyAndComposition present=%d fmt=%u size=%llux%u enableAutoReactive=%d colorOpaqueOnly present=%d autoTcThreshold=%.4f autoTcScale=%.4f autoReactiveScale=%.4f autoReactiveMax=%.4f",
                        r.diag,w,h,c->desc.displaySize.width,c->desc.displaySize.height,d->jitterOffset.x,d->jitterOffset.y,d->motionVectorScale.x,d->motionVectorScale.y,d->preExposure,d->frameTimeDelta,int(d->reset),unsigned(c->desc.flags),d->cameraNear,d->cameraFar,d->cameraFovAngleVertical,int(d->enableSharpening),d->sharpness,int(in.color.format),int(in.depth.format),int(in.motion.format),int(in.output.format),int(in.exposure.format),
                        int(d->reactive.resource!=nullptr),unsigned(reactive.Format),static_cast<unsigned long long>(reactive.Width),reactive.Height,
                        int(d->transparencyAndComposition.resource!=nullptr),unsigned(composition.Format),static_cast<unsigned long long>(composition.Width),composition.Height,
                        int(d->enableAutoReactive),int(d->colorOpaqueOnly.resource!=nullptr),d->autoTcThreshold,d->autoTcScale,d->autoReactiveScale,d->autoReactiveMax);
                    r.log(b); }
                // FSR 2.2.1 ffx_fsr2.cpp lines 905-912: cancellation = (previousJitter - jitter) / mvTargetSize,
                // applied in ffx_fsr2_callbacks_hlsl.h line 492 as "motionVector -= cancellation".
                if(c->desc.flags&FFX_FSR2_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION) {
                    in.jitter_cancel[0]=(c->jitter.x-d->jitterOffset.x)/mx;in.jitter_cancel[1]=(c->jitter.y-d->jitterOffset.y)/my;
                }
                c->pipeline->record(static_cast<ID3D12GraphicsCommandList*>(d->commandList),in);
                if(!c->ours) r.log("Learned 2x dispatch active");
                c->ours=true;c->jitter=d->jitterOffset; r.dump.drain(c->dumps,c->dispatches); return FFX_OK;
            }
            if(c->pipeline) c->pipeline->reset();
        }
    } catch(const FrameSkip& e) { if(!r.skip_logged) { r.skip_logged=true; r.log(e.what()); } }
    catch(const std::exception& e) { r.fail(e.what()); } catch(...) { r.fail("dispatch failed"); }
    if(c) { c->ours=false; if(readable(d,sizeof(*d))) c->jitter=d->jitterOffset; }
    FfxErrorCode result;
    if(reset_original && readable(d,sizeof(*d))) {
        auto copy=*d; copy.reset=true; result=r.ffxFsr2ContextDispatch(context,&copy);
        if(c && result==FFX_OK) r.dump.record(c->dumps,c->device.Get(),c->dispatches,r.ring_frames+2,copy,c->desc,previous_jitter,reinterpret_cast<uintptr_t>(context),resource);
    } else {
        result=r.ffxFsr2ContextDispatch(context,d);
        if(c && result==FFX_OK && readable(d,sizeof(*d))) r.dump.record(c->dumps,c->device.Get(),c->dispatches,r.ring_frames+2,*d,c->desc,previous_jitter,reinterpret_cast<uintptr_t>(context),resource);
    }
    if(result!=FFX_OK && r.dump.forwarding()) r.dump.abort("dump: original FSR 2 dispatch failed");
    if(c) r.dump.drain(c->dumps,c->dispatches);
    return result;
}
FfxErrorCode ffxFsr2ContextDestroy(FfxFsr2Context* c) {
    auto& r=runtime(); std::lock_guard<std::recursive_mutex> lock(r.mutex);
    auto it=r.contexts.find(c);
    if(it!=r.contexts.end()) r.dump.drain(it->second.dumps,it->second.dispatches,true);
    r.contexts.erase(c); return r.ffxFsr2ContextDestroy?r.ffxFsr2ContextDestroy(c):FFX_ERROR_BACKEND_API_ERROR;
}
bool ffxAssertReport(const char* file,int32_t line,const char* condition,const char* msg) { auto fn=runtime().ffxAssertReport;return fn?fn(file,line,condition,msg):true; }
void ffxAssertSetPrintingCallback(FfxAssertCallback cb) { auto fn=runtime().ffxAssertSetPrintingCallback;if(fn) fn(cb); }
FfxErrorCode ffxFsr2ContextGenerateReactiveMask(FfxFsr2Context* c,const FfxFsr2GenerateReactiveDescription* d) { auto fn=runtime().ffxFsr2ContextGenerateReactiveMask;return fn?fn(c,d):FFX_ERROR_BACKEND_API_ERROR; }
FfxErrorCode ffxFsr2GetJitterOffset(float* x,float* y,int32_t i,int32_t n) { auto fn=runtime().ffxFsr2GetJitterOffset;return fn?fn(x,y,i,n):FFX_ERROR_BACKEND_API_ERROR; }
int32_t ffxFsr2GetJitterPhaseCount(int32_t w,int32_t d) { auto fn=runtime().ffxFsr2GetJitterPhaseCount;return fn?fn(w,d):0; }
FfxErrorCode ffxFsr2GetRenderResolutionFromQualityMode(uint32_t* w,uint32_t* h,uint32_t dw,uint32_t dh,FfxFsr2QualityMode q) { auto fn=runtime().ffxFsr2GetRenderResolutionFromQualityMode;return fn?fn(w,h,dw,dh,q):FFX_ERROR_BACKEND_API_ERROR; }
float ffxFsr2GetUpscaleRatioFromQualityMode(FfxFsr2QualityMode q) { auto fn=runtime().ffxFsr2GetUpscaleRatioFromQualityMode;return fn?fn(q):0; }
bool ffxFsr2ResourceIsNull(FfxResource resource) { auto fn=runtime().ffxFsr2ResourceIsNull;return fn?fn(resource):!resource.resource; }
}
