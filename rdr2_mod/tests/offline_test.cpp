// Headless D3D12 harness: runs the exact Pipeline code used by the DLL on a sequence written by
// tools/export_sequence.py and compares every frame with the PyTorch reference.
// usage: offline_test weights.bin sequence.bin [--ring N] [--tol 3e-3] [--debug-view N] [--d2s-alt] [--sharpen S] [--stabilize S] [--stabilize-tau T] [--foliage S] [--fallback S] [--no-fast-path] [--dml-graph|--no-dml-graph] [--debug-layer] [--engine-space]
#include "gpu.h"
#include <d3d12sdklayers.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <windows.h>
static ID3D12Device* crash_device=nullptr;
static LONG WINAPI fsrm_crash(EXCEPTION_POINTERS* e) {
    if(e->ExceptionRecord->ExceptionCode!=EXCEPTION_ACCESS_VIOLATION) return EXCEPTION_CONTINUE_SEARCH;
    void* frames[32]; USHORT n=RtlCaptureStackBackTrace(0,32,frames,nullptr);
    auto show=[](const char* tag,void* a){ HMODULE m=nullptr; char name[MAX_PATH]="?";
        if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,(LPCSTR)a,&m)&&m) GetModuleFileNameA(m,name,MAX_PATH);
        const char* base=strrchr(name,'\\'); fprintf(stderr,"%s %s+0x%llx\n",tag,base?base+1:name,(unsigned long long)((char*)a-(char*)m)); };
    show("CRASH at",e->ExceptionRecord->ExceptionAddress);
    fprintf(stderr,"access %llu addr 0x%llx\n",(unsigned long long)e->ExceptionRecord->ExceptionInformation[0],(unsigned long long)e->ExceptionRecord->ExceptionInformation[1]);
    for(USHORT i=0;i<n;i++) show("  frame",frames[i]);
    if(crash_device) fprintf(stderr,"CRASH GetDeviceRemovedReason()=0x%08X\n",unsigned(crash_device->GetDeviceRemovedReason()));
    fsrmamba::drain_debug_messages(crash_device);
    fflush(stderr); ExitProcess(3); }

using namespace fsrmamba;
namespace {
struct Diagnostics {
    ComPtr<ID3D12Device> device;
    ~Diagnostics() {
        if(device) std::fprintf(stderr,"D3D12 exit: GetDeviceRemovedReason()=0x%08X\n",unsigned(device->GetDeviceRemovedReason()));
        drain_debug_messages(device.Get());
        crash_device=nullptr;
    }
};
bool enable_debug_layer() {
    const GUID iid={0x344488b7,0x6846,0x474b,{0xb9,0x89,0xf0,0x27,0x44,0x82,0x45,0xe0}};
    ComPtr<ID3D12Debug> debug;
    HRESULT hr=D3D12GetDebugInterface(iid,reinterpret_cast<void**>(debug.GetAddressOf()));
    if(FAILED(hr)) {
        std::fprintf(stderr,"D3D12 debug layer unavailable: 0x%08X; continuing without it (install Windows Graphics Tools)\n",unsigned(hr));
        return false;
    }
    debug->EnableDebugLayer();
    std::cout<<"D3D12 debug layer: enabled\n";
    return true;
}
void configure_info_queue(ID3D12Device* device) {
    const GUID iid={0x0742a90b,0xc387,0x483f,{0xb9,0x46,0x30,0xa7,0xe4,0xe6,0x14,0x58}};
    ComPtr<ID3D12InfoQueue> queue;
    HRESULT hr=device->QueryInterface(iid,reinterpret_cast<void**>(queue.GetAddressOf()));
    if(FAILED(hr)) {
        std::fprintf(stderr,"D3D12InfoQueue unavailable: 0x%08X; GetDeviceRemovedReason()=0x%08X\n",unsigned(hr),unsigned(device->GetDeviceRemovedReason()));
        drain_debug_messages(device);
        return;
    }
    queue->ClearStorageFilter(); queue->ClearRetrievalFilter();
    checked(queue->SetMessageCountLimit(UINT64_MAX),"Set debug message count limit",device);
}
void check(bool ok, const char* text) { if(!ok) throw std::runtime_error(text); }
template<class T> T get(std::istream& f) { T t{}; f.read(reinterpret_cast<char*>(&t), sizeof(t)); check(bool(f), "truncated sequence"); return t; }
std::vector<float> floats(std::istream& f, size_t n) {
    std::vector<float> v(n); f.read(reinterpret_cast<char*>(v.data()), n*4); check(bool(f), "truncated sequence"); return v;
}
struct Texture {
    ComPtr<ID3D12Resource> resource, upload; D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{}; UINT64 bytes=0; UINT rows=0; UINT64 row_bytes=0;
    DXGI_FORMAT format=DXGI_FORMAT_UNKNOWN; UINT w=0,h=0;
};
Texture make_texture(ID3D12Device* dev, UINT w, UINT h, DXGI_FORMAT format, bool uav, D3D12_RESOURCE_STATES state, bool with_upload) {
    Texture t; t.w=w; t.h=h; t.format=format;
    D3D12_HEAP_PROPERTIES props{}; props.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width=w; d.Height=h; d.DepthOrArraySize=1; d.MipLevels=1; d.Format=format; d.SampleDesc.Count=1;
    if(uav) d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    checked(dev->CreateCommittedResource(&props,D3D12_HEAP_FLAG_NONE,&d,state,nullptr,IID_PPV_ARGS(&t.resource)),"Create texture",dev);
    dev->GetCopyableFootprints(&d,0,1,0,&t.footprint,&t.rows,&t.row_bytes,&t.bytes);
    if(with_upload) t.upload=buffer(dev,t.bytes,D3D12_HEAP_TYPE_UPLOAD);
    return t;
}
template<class F> void fill(ID3D12Device* device, Texture& t, F value) { // value(x,y,out_bytes_ptr)
    uint8_t* mapped=nullptr; checked(t.upload->Map(0,nullptr,reinterpret_cast<void**>(&mapped)),"Map upload",device);
    for(UINT y=0;y<t.h;++y) value(y, mapped + t.footprint.Footprint.RowPitch*y);
    t.upload->Unmap(0,nullptr);
}
void copy_in(ID3D12GraphicsCommandList* list, Texture& t) {
    transition(list,t.resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{}; dst.pResource=t.resource.Get(); dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource=t.upload.Get(); src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; src.PlacedFootprint=t.footprint;
    list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    transition(list,t.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}
}
int main(int argc, char** argv) {
    AddVectoredExceptionHandler(1,fsrm_crash); setvbuf(stdout,nullptr,_IONBF,0);
    Diagnostics diagnostics;
    try {
        check(argc>=3, "usage: offline_test weights.bin sequence.bin [--ring N] [--tol T] [--debug-view N] [--d2s-alt] [--sharpen S] [--stabilize S] [--stabilize-tau T] [--foliage S] [--fallback S] [--no-fast-path] [--dml-graph|--no-dml-graph] [--debug-layer] [--engine-space]");
        PipelineOptions options; double tol=3e-3; uint32_t debug_view=0; bool sharpen=false,engine_space=false,debug_layer=false; float sharpness=0;
        float stabilize=0,stabilize_tau=.5f;
        FoliageControls foliage;
        auto number=[&](const char* text,float minimum,float maximum) {
            size_t used=0; const float value=std::stof(text,&used);
            check(used==std::strlen(text) && std::isfinite(value) && value>=minimum && value<=maximum,"numeric argument out of range or malformed");
            return value;
        };
        for(int i=3;i<argc;++i) {
            std::string a=argv[i];
            if(a=="--ring" && i+1<argc) options.ring_frames=std::stoul(argv[++i]);
            else if(a=="--tol" && i+1<argc) tol=std::stod(argv[++i]);
            else if(a=="--debug-view" && i+1<argc) debug_view=std::stoul(argv[++i]);
            else if(a=="--sharpen" && i+1<argc) { sharpen=true; sharpness=std::stof(argv[++i]); check(std::isfinite(sharpness)&&sharpness>=0&&sharpness<=1,"sharpness must be in [0,1]"); }
            else if(a=="--stabilize" && i+1<argc) stabilize=number(argv[++i],0,.95f);
            else if(a=="--stabilize-tau" && i+1<argc) stabilize_tau=number(argv[++i],.05f,4);
            else if(a=="--no-fast-path") options.fast_path=false;
            else if(a=="--no-dml-graph") options.dml_graph=false; else if(a=="--dml-graph") options.dml_graph=true;
            else if(a=="--debug-layer") debug_layer=true;
            else if(a=="--engine-space") engine_space=true;
            else if(a=="--d2s-alt") options.depth_to_space_alt=true;
            else {
                bool found=false;
                for(const auto& setting:foliage_settings) if(a==setting.cli && i+1<argc) {
                    foliage.*(setting.field)=number(argv[++i],setting.minimum,setting.maximum); found=true; break;
                }
                if(!found) throw std::runtime_error("unknown or incomplete argument "+a);
            }
        }
        set_log_sink([](const char* m){ std::cout<<"[log] "<<m<<'\n'; });
        wchar_t exe[MAX_PATH]; GetModuleFileNameW(nullptr,exe,MAX_PATH); set_directml_folder(std::filesystem::path(exe).parent_path());
        auto weights=Weights::read(argv[1]);
        check((stabilize==0 && !foliage.enabled()) || weights.config.kpn(),"temporal filters require KPN weights");
        std::cout<<"architecture "<<weights.config.arch<<"; max-abs tolerance "<<tol<<'\n';
        if(weights.config.kpn()) std::cout<<"KPN stride "<<weights.config.trunk_stride<<", taps "<<weights.config.taps
            <<", history "<<weights.config.history_filter<<"; one compiled graph, fp16 storage, shader shuffle\n";
        std::ifstream f(argv[2],std::ios::binary); check(bool(f),"cannot open sequence");
        char magic[8]; f.read(magic,8); check(f && std::memcmp(magic,"FSMSEQ1\0",8)==0,"bad sequence magic");
        const UINT w=get<uint32_t>(f), h=get<uint32_t>(f), frames=get<uint32_t>(f), meta=get<uint32_t>(f);
        check(w>0&&h>0&&w<=8192&&h<=8192&&frames>0&&meta<(1u<<20),"implausible sequence header");
        std::string metadata(meta,'\0'); f.read(metadata.data(),meta); check(bool(f),"truncated metadata");
        std::cout<<"sequence "<<w<<"x"<<h<<" -> "<<2*w<<"x"<<2*h<<", "<<frames<<" frames\nmetadata "<<metadata<<'\n';

        options.debug_layer=debug_layer && enable_debug_layer();
        auto& device=diagnostics.device;
        checked(D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)),"D3D12CreateDevice (no D3D12 adapter?)");
        crash_device=device.Get();
        if(options.debug_layer) configure_info_queue(device.Get());
        D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT; ComPtr<ID3D12CommandQueue> queue;
        checked(device->CreateCommandQueue(&qd,IID_PPV_ARGS(&queue)),"CreateCommandQueue",device.Get());
        ComPtr<ID3D12CommandAllocator> allocator; checked(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)),"CreateCommandAllocator",device.Get());
        ComPtr<ID3D12GraphicsCommandList> list; checked(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)),"CreateCommandList",device.Get()); checked(list->Close(),"initial list close",device.Get());
        ComPtr<ID3D12Fence> fence; checked(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"CreateFence",device.Get()); UINT64 fence_value=0;
        HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        if(!event) checked(HRESULT_FROM_WIN32(GetLastError()),"CreateEvent",device.Get());
        UINT64 frequency=0; checked(queue->GetTimestampFrequency(&frequency),"GetTimestampFrequency",device.Get());
        D3D12_QUERY_HEAP_DESC qh{D3D12_QUERY_HEAP_TYPE_TIMESTAMP,4,0}; ComPtr<ID3D12QueryHeap> heap; checked(device->CreateQueryHeap(&qh,IID_PPV_ARGS(&heap)),"CreateQueryHeap",device.Get());
        auto query_readback=buffer(device.Get(),64,D3D12_HEAP_TYPE_READBACK);

        auto color=make_texture(device.Get(),w,h,engine_space ? DXGI_FORMAT_R32G32B32A32_FLOAT:DXGI_FORMAT_R16G16B16A16_FLOAT,false,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,true);
        auto motion=make_texture(device.Get(),w,h,DXGI_FORMAT_R32G32_FLOAT,false,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,true);
        auto depth=make_texture(device.Get(),w,h,DXGI_FORMAT_R32_FLOAT,false,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,true);
        auto output=make_texture(device.Get(),2*w,2*h,DXGI_FORMAT_R16G16B16A16_FLOAT,true,D3D12_RESOURCE_STATE_COPY_SOURCE,false);
        auto readback=buffer(device.Get(),output.bytes,D3D12_HEAP_TYPE_READBACK);

        Pipeline pipeline(device.Get(),weights,w,h,options);
        std::cout<<"pipeline created\n";
        if(engine_space) std::cout<<"engine-space colour: inverse Reinhard input, Reinhard readback comparison; unit exposure\n";
        if(foliage.enabled()) std::cout<<"foliage "<<foliage.foliage_strength<<", fallback "<<foliage.fallback_strength<<"; reference errors show drift, parity comparison disabled\n";
        if(stabilize>0) std::cout<<"stabilize "<<stabilize<<", tau "<<stabilize_tau<<", eps 0.004; reference errors show drift, parity comparison disabled\n";
        if(sharpen) std::cout<<"RCAS sharpness "<<sharpness<<"; resolve_ms includes RCAS; reference comparison disabled\nframe   pack_ms  net_ms  resolve_ms  verdict\n";
        else std::cout<<"frame   max_err      mean_err     pack_ms  net_ms  resolve_ms  verdict\n";
        double worst=0; bool pass=true; std::vector<double> t_pack,t_net,t_res;
        for(UINT i=0;i<frames;++i) {
            const double jx=get<double>(f), jy=get<double>(f); const uint32_t first=get<uint32_t>(f);
            auto lr=floats(f,size_t(w)*h*3), mv=floats(f,size_t(w)*h*2), dp=floats(f,size_t(w)*h), ref=floats(f,size_t(4)*w*h*3);
            if(engine_space) fill(device.Get(),color,[&](UINT y,uint8_t* row){
                auto* o=reinterpret_cast<float*>(row);
                for(UINT x=0;x<w;++x) {
                    for(int c=0;c<3;++c) {
                        float v=from_half(to_half(lr[(size_t(y)*w+x)*3+c]));
                        check(std::isfinite(v) && v>=0 && v<1,"engine-space colour must be in [0,1) after fp16 rounding");
                        o[4*x+c]=v/(1-v);
                    }
                    o[4*x+3]=1;
                }
            });
            else fill(device.Get(),color,[&](UINT y,uint8_t* row){ auto* o=reinterpret_cast<uint16_t*>(row); for(UINT x=0;x<w;++x){ for(int c=0;c<3;++c) o[4*x+c]=to_half(lr[(size_t(y)*w+x)*3+c]); o[4*x+3]=0x3c00; } });
            fill(device.Get(),motion,[&](UINT y,uint8_t* row){ std::memcpy(row,&mv[size_t(y)*w*2],size_t(w)*8); });
            fill(device.Get(),depth,[&](UINT y,uint8_t* row){ std::memcpy(row,&dp[size_t(y)*w],size_t(w)*4); });
            checked(allocator->Reset(),"allocator reset",device.Get()); checked(list->Reset(allocator.Get(),nullptr),"list reset",device.Get());
            copy_in(list.Get(),color); copy_in(list.Get(),motion); copy_in(list.Get(),depth);
            FrameInputs in;
            in.color={color.resource.Get(),color.format,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
            in.motion={motion.resource.Get(),motion.format,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
            in.depth={depth.resource.Get(),depth.format,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
            in.output={output.resource.Get(),output.format,D3D12_RESOURCE_STATE_COPY_SOURCE};
            // The sequence stores model-space inputs and the jitter exactly as step_reference received it.
            in.jitter_x=jx; in.jitter_y=jy; in.reset=first!=0; in.model_space=!engine_space; in.inverted_depth=true; in.display_motion=false;
            in.debug_view=debug_view;in.sharpen=sharpen;in.sharpness=sharpness;
            in.foliage=foliage;in.stabilize=stabilize;in.stabilize_tau=stabilize_tau;
            pipeline.record(list.Get(),in,heap.Get());
            list->ResolveQueryData(heap.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0,4,query_readback.Get(),0);
            D3D12_TEXTURE_COPY_LOCATION dst{}, src{}; dst.pResource=readback.Get(); dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint=output.footprint;
            src.pResource=output.resource.Get(); src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
            checked(list->Close(),"list close",device.Get());
            ID3D12CommandList* lists[]={list.Get()}; queue->ExecuteCommandLists(1,lists);
            checked(queue->Signal(fence.Get(),++fence_value),"Signal",device.Get()); checked(fence->SetEventOnCompletion(fence_value,event),"SetEventOnCompletion",device.Get());
            const DWORD wait=WaitForSingleObject(event,60000);
            if(wait==WAIT_FAILED) checked(HRESULT_FROM_WIN32(GetLastError()),"WaitForSingleObject",device.Get());
            checked(device->GetDeviceRemovedReason(),"device removed",device.Get());
            check(wait==WAIT_OBJECT_0 && fence->GetCompletedValue()>=fence_value,"GPU did not finish (timeout)");
            uint8_t* mapped=nullptr; D3D12_RANGE rr{0,static_cast<SIZE_T>(output.bytes)}; checked(readback->Map(0,&rr,reinterpret_cast<void**>(&mapped)),"Map readback",device.Get());
            double max_err=0,sum=0; size_t n=0; bool finite=true;
            for(UINT y=0;y<2*h;++y) for(UINT x=0;x<2*w;++x) {
                const auto* px=reinterpret_cast<const uint16_t*>(mapped+output.footprint.Footprint.RowPitch*y)+4*x;
                for(int c=0;c<3;++c) {
                    float v=from_half(px[c]); if(!std::isfinite(v)) finite=false;
                    if(engine_space && std::isfinite(v)) v=v/(1+v);
                    if(!sharpen) { double e=std::abs(double(v)-ref[(size_t(y)*2*w+x)*3+c]); max_err=std::max(max_err,e); sum+=e; ++n; }
                }
            }
            D3D12_RANGE none{0,0}; readback->Unmap(0,&none);
            uint64_t* ts=nullptr; D3D12_RANGE qr{0,32}; checked(query_readback->Map(0,&qr,reinterpret_cast<void**>(&ts)),"Map queries",device.Get());
            const double ms[3]={1e3*double(ts[1]-ts[0])/frequency,1e3*double(ts[2]-ts[1])/frequency,1e3*double(ts[3]-ts[2])/frequency}; query_readback->Unmap(0,&none);
            const bool compare=!debug_view && !sharpen && stabilize==0 && !foliage.enabled();
            const bool ok=finite && (!compare || max_err<=tol); pass=pass&&ok; worst=std::max(worst,max_err);
            if(i>0) { t_pack.push_back(ms[0]); t_net.push_back(ms[1]); t_res.push_back(ms[2]); }
            if(sharpen) std::printf("%5u   %7.3f %7.3f %9.3f   %s\n",i,ms[0],ms[1],ms[2],finite?"finite":"NON-FINITE");
            else std::printf("%5u   %.3e   %.3e   %7.3f %7.3f %9.3f   %s\n",i,max_err,sum/n,ms[0],ms[1],ms[2],finite?(compare?(ok?"ok":"ABOVE TOLERANCE"):"not compared"):"NON-FINITE");
        }
        auto avg=[](std::vector<double>& v){ double s=0; for(double x:v) s+=x; return v.empty()?0.0:s/v.size(); };
        std::printf("mean GPU ms (frames 1..N): pack %.3f  network %.3f  resolve %.3f  total %.3f\n",avg(t_pack),avg(t_net),avg(t_res),avg(t_pack)+avg(t_net)+avg(t_res));
        if(sharpen) std::printf("RCAS benchmark: %s\n",pass?"finite output":"NON-FINITE");
        else if(stabilize>0 || foliage.enabled()) std::printf("temporal filter benchmark: %s; worst reference drift %.3e (not parity)\n",pass?"finite output":"NON-FINITE",worst);
        else std::printf("worst max abs error %.3e (tolerance %.1e): %s\n",worst,tol,debug_view?"debug view, not compared":(pass?"PASS":"FAIL"));
        return pass?0:2;
    } catch(const std::exception& e) {
        std::cerr<<"offline_test failed: "<<e.what()<<'\n';
        if(diagnostics.device) std::fprintf(stderr,"D3D12 failure: GetDeviceRemovedReason()=0x%08X\n",unsigned(diagnostics.device->GetDeviceRemovedReason()));
        drain_debug_messages(diagnostics.device.Get());
        return 1;
    }
}
