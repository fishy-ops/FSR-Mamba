#pragma once
#include "gpu.h"
#include "ffx_fsr2.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace fsrmamba {
inline D3D12_RESOURCE_DESC dump_resource_desc(ID3D12Resource* resource) {
#if defined(__MINGW32__)
    // MS COM ABI: this precedes the hidden struct return pointer (vtable slot 10).
    using Fn=D3D12_RESOURCE_DESC*(*)(ID3D12Resource*,D3D12_RESOURCE_DESC*);
    D3D12_RESOURCE_DESC out{}; (*reinterpret_cast<Fn**>(resource))[10](resource,&out); return out;
#else
    return resource->GetDesc();
#endif
}
struct DumpPlane {
    const char* name;
    InputResource source;
    ComPtr<ID3D12Resource> readback;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    uint64_t bytes=0,offset=0;
    UINT rows=0;
    DXGI_FORMAT format=DXGI_FORMAT_UNKNOWN;
};
struct DumpFrame {
    uint64_t ready=0,bytes=0;
    std::filesystem::path path;
    std::string metadata;
    std::vector<DumpPlane> planes;
};
struct DumpQueue { std::deque<DumpFrame> frames; };
class DumpRun {
    static constexpr uint64_t limit=60000000000ull,minimum_free=25000000000ull;
    uint32_t requested=0,every=1,captured=0,written=0;
    uint64_t seen=0,index=0,total=0,reserved=0;
    bool active=false,failed=false;
    std::filesystem::path directory;
    void* owner;
    void (*sink)(void*,const char*) noexcept;
    void log(const char* message) const noexcept { sink(owner,message); }
    void space(uint64_t bytes=0) {
        ULARGE_INTEGER available{};
        if(!GetDiskFreeSpaceExW(directory.c_str(),&available,nullptr,nullptr)) throw std::runtime_error("dump: GetDiskFreeSpaceExW failed");
        if(available.QuadPart<minimum_free+bytes) throw std::runtime_error("dump: insufficient free space to retain 25 GB");
    }
    void stop(const char* message) noexcept {
        active=false;
        if(!failed) { failed=true; try { log(message); } catch(...) {} }
    }
    static void scalar(std::ostream& out,float value) {
        if(std::isfinite(value)) out<<value; else out<<"null";
    }
    static std::string metadata(const FfxFsr2DispatchDescription& d,const FfxFsr2ContextDescription& c,const std::vector<DumpPlane>& planes,const FfxFloatCoords2D& previous_jitter,uint64_t context_id) {
        std::ostringstream out; out.imbue(std::locale::classic()); out<<std::setprecision(9);
        out<<"{\"renderSize\":["<<d.renderSize.width<<','<<d.renderSize.height<<"],\"displaySize\":["<<c.displaySize.width<<','<<c.displaySize.height<<"],\"jitterOffset\":[";
        scalar(out,d.jitterOffset.x); out<<','; scalar(out,d.jitterOffset.y);
        out<<"],\"previousJitterOffset\":["; scalar(out,previous_jitter.x); out<<','; scalar(out,previous_jitter.y);
        out<<"],\"context_id\":"<<context_id<<",\"motionVectorScale\":["; scalar(out,d.motionVectorScale.x); out<<','; scalar(out,d.motionVectorScale.y); out<<']';
        const char* names[]={"preExposure","frameTimeDelta","cameraNear","cameraFar","cameraFovAngleVertical","sharpness"};
        const float values[]={d.preExposure,d.frameTimeDelta,d.cameraNear,d.cameraFar,d.cameraFovAngleVertical,d.sharpness};
        for(int i=0;i<6;++i) { out<<",\""<<names[i]<<"\":"; scalar(out,values[i]); }
        out<<",\"enableAutoReactive\":"<<(d.enableAutoReactive?"true":"false")<<",\"colorOpaqueOnly_present\":"<<(d.colorOpaqueOnly.resource?"true":"false");
        const char* auto_names[]={"autoTcThreshold","autoTcScale","autoReactiveScale","autoReactiveMax"};
        const float auto_values[]={d.autoTcThreshold,d.autoTcScale,d.autoReactiveScale,d.autoReactiveMax};
        for(int i=0;i<4;++i) { out<<",\""<<auto_names[i]<<"\":"; scalar(out,auto_values[i]); }
        out<<",\"reset\":"<<(d.reset?"true":"false")<<",\"enableSharpening\":"<<(d.enableSharpening?"true":"false")<<",\"context_flags\":"<<c.flags<<",\"planes\":[";
        bool comma=false;
        for(const auto& p:planes) {
            if(comma) out<<',';
            comma=true;
            out<<"{\"name\":\""<<p.name<<"\",\"dxgi_format\":"<<unsigned(p.format)<<",\"footprint_format\":"<<unsigned(p.footprint.Footprint.Format)<<",\"width\":"<<p.footprint.Footprint.Width<<",\"height\":"<<p.rows
               <<",\"row_pitch\":"<<p.footprint.Footprint.RowPitch<<",\"offset\":"<<p.offset<<",\"bytes\":"<<p.bytes<<'}';
        }
        out<<"]}"; return out.str();
    }
public:
    DumpRun(void* owner,void (*sink)(void*,const char*) noexcept) noexcept:owner(owner),sink(sink) {}
    bool forwarding() const noexcept { return active || failed; }
    void start(uint32_t n,uint32_t k,const std::filesystem::path& dir) noexcept {
        if(active || failed || !n) return;
        try {
            directory=dir; std::filesystem::create_directories(directory); space();
            requested=n; every=std::max(k,1u); captured=written=0; seen=0; active=true;
            log("dump started: forwarding to original FSR 2");
        } catch(const std::exception& e) { stop(e.what()); } catch(...) { stop("dump: initialization failed"); }
    }
    void cancel() noexcept {
        if(!active) return;
        requested=captured;
        if(written==captured) {
            active=false;
            try { log("dump stopped"); } catch(...) {}
        }
    }
    void abort(const char* message) noexcept { stop(message); }
    template<class Resource>
    void record(DumpQueue& queue,ID3D12Device* device,uint64_t dispatch,uint32_t delay,
                const FfxFsr2DispatchDescription& d,const FfxFsr2ContextDescription& c,
                const FfxFloatCoords2D& previous_jitter,uint64_t context_id,Resource resource) noexcept {
        if(!active || captured>=requested || ++seen%every) return;
        try {
            space();
            auto* list=static_cast<ID3D12GraphicsCommandList*>(d.commandList);
            if(!device || !list || list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT) throw std::runtime_error("dump: direct command list required");
            DumpFrame frame; frame.ready=dispatch+delay;
            const char* names[]={"color","depth","motionVectors","exposure","output","reactive","transparencyAndComposition"};
            const FfxResource* sources[]={&d.color,&d.depth,&d.motionVectors,&d.exposure,&d.output,&d.reactive,&d.transparencyAndComposition};
            for(int i=0;i<7;++i) {
                if((i==3 || i>=5) && !sources[i]->resource) continue;
                DumpPlane p{}; p.name=names[i]; p.source=resource(*sources[i]);
                if(!p.source.resource) throw std::runtime_error("dump: required resource missing");
                const auto desc=dump_resource_desc(p.source.resource); p.format=desc.Format;
                if((desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE1D) || desc.SampleDesc.Count!=1)
                    throw std::runtime_error("dump: unsupported resource dimension or MSAA");
                UINT64 row_bytes=0,size=0;
                // Subresource zero is mip zero, array slice zero, depth plane zero.
                device->GetCopyableFootprints(&desc,0,1,0,&p.footprint,&p.rows,&row_bytes,&size);
                p.bytes=uint64_t(p.footprint.Footprint.RowPitch)*p.rows; p.offset=frame.bytes;
                if(!p.rows || !p.bytes || size==UINT64_MAX || p.footprint.Footprint.Depth!=1 || p.bytes>limit-frame.bytes)
                    throw std::runtime_error("dump: invalid copy footprint");
                p.readback=buffer(device,std::max(size,p.bytes+p.footprint.Offset),D3D12_HEAP_TYPE_READBACK);
                frame.bytes+=p.bytes; frame.planes.push_back(std::move(p));
            }
            frame.metadata=metadata(d,c,frame.planes,previous_jitter,context_id);
            frame.metadata.pop_back(); frame.metadata+=",\"dispatch_index\":"+std::to_string(dispatch)+",\"dump_every\":"+std::to_string(every)+"}";
            frame.bytes+=12+frame.metadata.size();
            if(frame.bytes>limit-total-reserved) throw std::runtime_error("dump: 60 GB byte limit reached");
            space(frame.bytes);
            do {
                std::wostringstream name; name<<L"frame_"<<std::setfill(L'0')<<std::setw(5)<<index++<<L".bin";
                frame.path=directory/name.str();
            } while(std::filesystem::exists(frame.path));
            // Queue ownership and all allocations precede commands referencing the readbacks.
            queue.frames.push_back(std::move(frame)); auto& queued=queue.frames.back();
            reserved+=queued.bytes; ++captured;
            for(auto& p:queued.planes) {
                transition(list,p.source.resource,p.source.state,D3D12_RESOURCE_STATE_COPY_SOURCE);
                D3D12_TEXTURE_COPY_LOCATION src{},dst{};
                src.pResource=p.source.resource; src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX; src.SubresourceIndex=0;
                dst.pResource=p.readback.Get(); dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT; dst.PlacedFootprint=p.footprint;
                list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
                transition(list,p.source.resource,D3D12_RESOURCE_STATE_COPY_SOURCE,p.source.state);
            }
        } catch(const std::exception& e) { stop(e.what()); } catch(...) { stop("dump: recording failed"); }
    }
    void drain(DumpQueue& queue,uint64_t dispatch,bool destroy=false) noexcept {
        while(!queue.frames.empty() && queue.frames.front().ready<=dispatch) {
            auto& frame=queue.frames.front();
            try {
                if(!failed) {
                    space(frame.bytes);
                    std::ofstream out(frame.path,std::ios::binary); out.exceptions(std::ios::badbit|std::ios::failbit);
                    out.write("FSMDUMP1",8); const uint32_t length=static_cast<uint32_t>(frame.metadata.size());
                    const char le[]={char(length),char(length>>8),char(length>>16),char(length>>24)};
                    out.write(le,4); out.write(frame.metadata.data(),frame.metadata.size());
                    for(auto& p:frame.planes) {
                        void* data=nullptr; D3D12_RANGE range{SIZE_T(p.footprint.Offset),SIZE_T(p.footprint.Offset+p.bytes)};
                        checked(p.readback->Map(0,&range,&data),"dump: Map readback");
                        D3D12_RANGE empty{0,0};
                        try { out.write(static_cast<const char*>(data)+p.footprint.Offset,p.bytes); }
                        catch(...) { p.readback->Unmap(0,&empty); throw; }
                        p.readback->Unmap(0,&empty);
                    }
                    out.close(); total+=frame.bytes; ++written;
                    if(written==requested) {
                        active=false;
                        log(("dump complete: "+std::to_string(written)+" frames in "+directory.u8string()).c_str());
                    }
                }
            } catch(const std::exception& e) {
                std::error_code ec; std::filesystem::remove(frame.path,ec); stop(e.what());
            } catch(...) { stop("dump: writing failed"); }
            reserved-=frame.bytes; queue.frames.pop_front();
        }
        if(destroy && !queue.frames.empty()) {
            const char* message="dump: context destroyed before ring_frames + 2 dispatches; dropping pending frames";
            if(failed) log(message); else stop(message);
            for(const auto& f:queue.frames) reserved-=f.bytes;
            queue.frames.clear();
        }
    }
};
}
