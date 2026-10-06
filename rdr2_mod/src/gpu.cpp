#include "gpu.h"
#define DML_TARGET_VERSION 0x3000
#include <DirectML.h>
#include <d3d12sdklayers.h>
#ifdef __MINGW32__
// mingw has no __declspec(uuid): supply the DirectML interface IDs (values from DirectML.h).
#define FSRM_UUID(T,a,b,c,d0,d1,d2,d3,d4,d5,d6,d7) template<> const GUID& __mingw_uuidof<T>() { static const GUID g={a,b,c,{d0,d1,d2,d3,d4,d5,d6,d7}}; return g; }
FSRM_UUID(IDMLDevice1,0xa0884f9a,0xd2be,0x4355,0xaa,0x5d,0x59,0x01,0x28,0x1a,0xd1,0xd2)
FSRM_UUID(IDMLDevice,0x6dbd6437,0x96fd,0x423f,0xa9,0x8c,0xae,0x5e,0x7c,0x2a,0x57,0x3f)
FSRM_UUID(IDMLOperator,0x26caae7a,0x3081,0x4633,0x95,0x81,0x22,0x6f,0xbe,0x57,0x69,0x5d)
FSRM_UUID(IDMLCompiledOperator,0x6b15e56a,0xbf5c,0x4902,0x92,0xd8,0xda,0x3a,0x65,0x0a,0xfe,0xa4)
FSRM_UUID(IDMLOperatorInitializer,0x427c1113,0x435c,0x469c,0x86,0x76,0x4d,0x5d,0xd0,0x72,0xf8,0x13)
FSRM_UUID(IDMLBindingTable,0x29c687dc,0xde74,0x4e3b,0xab,0x00,0x11,0x68,0xf2,0xfc,0x3c,0xfc)
FSRM_UUID(IDMLCommandRecorder,0xe6857a76,0x2e3e,0x4fdd,0xbf,0xf4,0x5d,0x2b,0xa1,0x0f,0xb4,0x53)
#endif
#include "pack_dxil.h"
#include "resolve_dxil.h"
#include "pack_fast_dxil.h"
#include "resolve_fast_dxil.h"
#include "kpn_pack_dxil.h"
#include "kpn_resolve_dxil.h"
#include "kpn_resolve3_dxil.h"
#include "exposure_dxil.h"
#include "rcas_dxil.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <winver.h>

namespace fsrmamba {
namespace {
void (*g_sink)(const char*)=nullptr;
std::filesystem::path g_dml_folder;
void note(const std::string& s) { if(g_sink) g_sink(s.c_str()); }
HMODULE load_directml() {
    static HMODULE module=nullptr; static bool tried=false;
    if(tried) return module;
    tried=true;
    if(!g_dml_folder.empty()) {
        auto path=g_dml_folder/L"DirectML.dll";
        if(std::filesystem::exists(path)) module=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if(!module) note("DirectML.dll beside the proxy not loadable; trying System32");
    }
    if(!module) module=LoadLibraryExW(L"DirectML.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    if(!module) return nullptr;
    wchar_t where[MAX_PATH*2]; DWORD n=GetModuleFileNameW(module,where,MAX_PATH*2);
    std::string text="DirectML loaded from "; for(DWORD i=0;i<n;++i) text+=where[i]<128?char(where[i]):'?';
    DWORD ignored=0; DWORD size=n?GetFileVersionInfoSizeW(where,&ignored):0;
    if(size) {
        std::vector<char> data(size); VS_FIXEDFILEINFO* info=nullptr; UINT len=0;
        if(GetFileVersionInfoW(where,0,size,data.data()) && VerQueryValueW(data.data(),L"\\",reinterpret_cast<void**>(&info),&len) && info)
            text+=" version "+std::to_string(HIWORD(info->dwFileVersionMS))+"."+std::to_string(LOWORD(info->dwFileVersionMS))+"."+std::to_string(HIWORD(info->dwFileVersionLS))+"."+std::to_string(LOWORD(info->dwFileVersionLS));
    }
    note(text); return module;
}
}
void set_log_sink(void (*sink)(const char*)) { g_sink=sink; }
void set_directml_folder(const std::filesystem::path& folder) { g_dml_folder=folder; }
void drain_debug_messages(ID3D12Device* device) noexcept {
    if(!device) return;
    // Explicit IID also works with headers which lack __uuidof support.
    const GUID iid={0x0742a90b,0xc387,0x483f,{0xb9,0x46,0x30,0xa7,0xe4,0xe6,0x14,0x58}};
    ComPtr<ID3D12InfoQueue> queue;
    if(FAILED(device->QueryInterface(iid,reinterpret_cast<void**>(queue.GetAddressOf())))) return;
    queue->ClearRetrievalFilter();
    const auto count=queue->GetNumStoredMessages();
    try {
        for(UINT64 i=0;i<count;++i) {
            SIZE_T size=0; HRESULT hr=queue->GetMessage(i,nullptr,&size);
            if(FAILED(hr)) { std::fprintf(stderr,"D3D12InfoQueue GetMessage(%llu) size failed: 0x%08X\n",(unsigned long long)i,unsigned(hr)); continue; }
            std::vector<uint8_t> bytes(size);
            auto* message=reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
            hr=queue->GetMessage(i,message,&size);
            if(FAILED(hr)) { std::fprintf(stderr,"D3D12InfoQueue GetMessage(%llu) failed: 0x%08X\n",(unsigned long long)i,unsigned(hr)); continue; }
            const char* severity="UNKNOWN";
            switch(message->Severity) {
                case D3D12_MESSAGE_SEVERITY_CORRUPTION: severity="CORRUPTION"; break;
                case D3D12_MESSAGE_SEVERITY_ERROR: severity="ERROR"; break;
                case D3D12_MESSAGE_SEVERITY_WARNING: severity="WARNING"; break;
                case D3D12_MESSAGE_SEVERITY_INFO: severity="INFO"; break;
                case D3D12_MESSAGE_SEVERITY_MESSAGE: severity="MESSAGE"; break;
            }
            std::fprintf(stderr,"D3D12InfoQueue severity=%s(%u) id=%u: %s\n",severity,unsigned(message->Severity),unsigned(message->ID),message->pDescription ? message->pDescription:"");
        }
        if(auto discarded=queue->GetNumMessagesDiscardedByMessageCountLimit())
            std::fprintf(stderr,"D3D12InfoQueue discarded %llu messages at the storage limit\n",(unsigned long long)discarded);
        queue->ClearStoredMessages();
    } catch(...) { std::fprintf(stderr,"D3D12InfoQueue drain incomplete (allocation failed)\n"); }
    std::fflush(stderr);
}
void checked(HRESULT hr,const char* op,ID3D12Device* device) {
    if(SUCCEEDED(hr)) return;
    char codes[128];
    if(device) std::snprintf(codes,sizeof(codes),": HRESULT 0x%08X; GetDeviceRemovedReason()=0x%08X",unsigned(hr),unsigned(device->GetDeviceRemovedReason()));
    else std::snprintf(codes,sizeof(codes),": HRESULT 0x%08X",unsigned(hr));
    drain_debug_messages(device);
    throw std::runtime_error(std::string(op)+codes);
}
ComPtr<ID3D12Resource> buffer(ID3D12Device* dev,uint64_t bytes,D3D12_HEAP_TYPE heap) {
    D3D12_HEAP_PROPERTIES props{}; props.Type=heap;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; desc.Width=(bytes+3)&~3ULL;
    desc.Height=1; desc.DepthOrArraySize=1; desc.MipLevels=1; desc.SampleDesc.Count=1; desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if(heap==D3D12_HEAP_TYPE_DEFAULT) desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    auto state=heap==D3D12_HEAP_TYPE_UPLOAD?D3D12_RESOURCE_STATE_GENERIC_READ:heap==D3D12_HEAP_TYPE_READBACK?D3D12_RESOURCE_STATE_COPY_DEST:D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> out; checked(dev->CreateCommittedResource(&props,D3D12_HEAP_FLAG_NONE,&desc,state,nullptr,IID_PPV_ARGS(&out)),"Create buffer",dev); return out;
}
void transition(ID3D12GraphicsCommandList* list,ID3D12Resource* r,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after) {
    if(before==after) return;
    D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION; b.Transition={r,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after}; list->ResourceBarrier(1,&b);
}
void uav_barrier(ID3D12GraphicsCommandList* list) { D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV; list->ResourceBarrier(1,&b); }
namespace {
using Shape=std::array<UINT,4>;
uint64_t tensor_bytes(Shape shape) { uint64_t n=2; for(auto d:shape) { if(!d || n>(1ULL<<32)/d) throw std::runtime_error("tensor allocation limit"); n*=d; } return (n+3)&~3ULL; }
struct TensorDesc {
    Shape sizes;
    DML_BUFFER_TENSOR_DESC buffer_desc{};
    DML_TENSOR_DESC desc{};
    explicit TensorDesc(Shape shape,bool owned=false):sizes(shape) {
        buffer_desc.Flags=owned ? DML_TENSOR_FLAG_OWNED_BY_DML:DML_TENSOR_FLAG_NONE;
        buffer_desc.DataType=DML_TENSOR_DATA_TYPE_FLOAT16; buffer_desc.DimensionCount=4; buffer_desc.Sizes=sizes.data();
        buffer_desc.TotalTensorSizeInBytes=tensor_bytes(shape); desc={DML_TENSOR_TYPE_BUFFER,&buffer_desc};
    }
};
struct GpuTensor { Shape shape; ComPtr<ID3D12Resource> resource; bool owned=false; };
ComPtr<ID3D12DescriptorHeap> heap(ID3D12Device* dev,UINT count) {
    D3D12_DESCRIPTOR_HEAP_DESC desc{}; desc.NumDescriptors=std::max(count,1u); desc.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV; desc.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> out; HRESULT hr=dev->CreateDescriptorHeap(&desc,IID_PPV_ARGS(&out));
    auto operation="Create descriptor heap count "+std::to_string(count); checked(hr,operation.c_str(),dev);
    return out;
}
ComPtr<ID3D12Resource> texture(ID3D12Device* dev,UINT w,UINT h,DXGI_FORMAT format,D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES props{}; props.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; d.Width=w; d.Height=h; d.DepthOrArraySize=1; d.MipLevels=1; d.Format=format; d.SampleDesc.Count=1; d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    ComPtr<ID3D12Resource> r; checked(dev->CreateCommittedResource(&props,D3D12_HEAP_FLAG_NONE,&d,state,nullptr,IID_PPV_ARGS(&r)),"Create history texture",dev); return r;
}
ComPtr<ID3D12Resource> upload(ID3D12Device* dev,const void* data,size_t bytes) {
    auto r=buffer(dev,bytes,D3D12_HEAP_TYPE_UPLOAD); void* mapped=nullptr; D3D12_RANGE empty{0,0}; checked(r->Map(0,&empty,&mapped),"Map upload",dev);
    std::memset(mapped,0,static_cast<size_t>(r->GetDesc().Width)); std::memcpy(mapped,data,bytes); r->Unmap(0,nullptr); return r;
}
struct Operator {
    ComPtr<IDMLCompiledOperator> compiled;
    ComPtr<IDMLOperatorInitializer> initializer;
    ComPtr<IDMLBindingTable> binding,init_binding;
    ComPtr<ID3D12DescriptorHeap> descriptors,init_descriptors;
    ComPtr<ID3D12Resource> persistent,temp,init_temp;
};
// IDMLDispatchable::GetBindingProperties returns a struct by value. MSVC's member-function ABI passes
// `this` first and the hidden return pointer second; mingw swaps them, so call vtable slot 8 explicitly.
DML_BINDING_PROPERTIES binding_properties(IDMLDispatchable* d) {
#if defined(__MINGW32__)
    using Fn=DML_BINDING_PROPERTIES*(STDMETHODCALLTYPE*)(IDMLDispatchable*,DML_BINDING_PROPERTIES*);
    DML_BINDING_PROPERTIES out{}; (*reinterpret_cast<Fn**>(d))[8](d,&out); return out;
#else
    return d->GetBindingProperties();
#endif
}
// MinGW's D3D12 headers use WIDL_EXPLICIT_AGGREGATE_RETURNS for GetDesc and heap handles;
// their inline wrappers already pass this before the return pointer, unlike DirectML.h.
DML_BUFFER_BINDING buffer_binding(ID3D12Resource* resource,uint64_t required) {
    if(!resource) throw std::runtime_error("missing DirectML buffer");
    auto desc=resource->GetDesc();
    if(desc.Dimension!=D3D12_RESOURCE_DIMENSION_BUFFER || desc.Width<required || desc.Width%4)
        throw std::runtime_error("DirectML buffer size/layout mismatch: required "+std::to_string(required)+", width "+std::to_string(desc.Width));
    return {resource,0,desc.Width};
}
void bind_resource(IDMLBindingTable* table,ID3D12Resource* resource,uint64_t required,bool persistent) {
    if(!resource) return;
    auto b=buffer_binding(resource,required); DML_BINDING_DESC desc{DML_BINDING_TYPE_BUFFER,&b};
    if(persistent) table->BindPersistentResource(&desc); else table->BindTemporaryResource(&desc);
}
}
struct Pipeline::Impl {
    ComPtr<ID3D12Device> device;
    ComPtr<IDMLDevice> dml;
    ComPtr<IDMLDevice1> dml1;
    ComPtr<IDMLCommandRecorder> recorder;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pack,resolve,pack_fast,resolve_fast,kpn_pack,kpn_resolve,kpn_resolve3,exposure,rcas;
    Weights weights;
    FrameMaths math; bool d2s_alt=false,fast_path=true,graph_mode=false;
    struct Node { ComPtr<IDMLOperator> raw; std::vector<size_t> inputs; size_t output; std::string name; };
    std::vector<Node> nodes;
    Operator graph;
    UINT w,h,wp,hp; bool first=true,initialized=false,stabilizing=false; int ping=0;
    std::vector<Operator> ops;
    std::vector<GpuTensor> tensors;
    std::map<std::string,size_t> parameters;
    std::map<size_t,std::vector<uint8_t>> initial_data;
    size_t input,output,head_weight,head_bias;
    struct Slot {
        ComPtr<ID3D12Resource> data;
        ComPtr<ID3D12DescriptorHeap> descriptors;
        uint8_t* mapped=nullptr;
    };
    std::vector<Slot> slots;
    uint64_t frame=0, bias_offset=0;
    std::map<size_t,ComPtr<ID3D12Resource>> initial_uploads;
    ComPtr<ID3D12Resource> hcl,hraw,cw,reset_buffer;
    FoliageControls last_foliage;
    float last_pre_exposure=1;
    bool last_model_space=true,last_exposure_one=true;
    ComPtr<ID3D12Resource> tracker[2];
    ComPtr<ID3D12Resource> history[2],depth[2],coverage_state[2],age_state[2],auto_exposure,display;
    bool exposure_ready=false;
    void check_dml(const char* operation) {
        checked(device->GetDeviceRemovedReason(),operation,device.Get());
        checked(dml->GetDeviceRemovedReason(),operation,device.Get());
    }
    size_t tensor(Shape shape) {
        tensor_bytes(shape);
        tensors.push_back({shape,graph_mode ? ComPtr<ID3D12Resource>{}:buffer(device.Get(),tensor_bytes(shape),D3D12_HEAP_TYPE_DEFAULT)});
        return tensors.size()-1;
    }
    size_t parameter(const std::string& name,bool bias=false) {
        auto found=parameters.find(name); if(found!=parameters.end()) return found->second;
        const auto& t=weights.at(name); Shape shape{};
        if(bias) shape={1,t.shape[0],1,1}; else std::copy(t.shape.begin(),t.shape.end(),shape.begin());
        size_t id=tensor(shape); parameters[name]=id;
        tensors[id].owned=graph_mode && t.dtype==1;
        if(t.dtype==1) initial_data[id]=t.bytes;
        return id;
    }
    void op(DML_OPERATOR_TYPE type,const void* desc,const std::vector<size_t>& ins,size_t out) {
        Operator o; DML_OPERATOR_DESC d{type,desc}; ComPtr<IDMLOperator> raw;
        const auto name="node_"+std::to_string(graph_mode ? nodes.size():ops.size())+"_type_"+std::to_string(int(type));
        { std::string what="Create DirectML operator "+name+" out";
          for(auto v:tensors[out].shape) what+=" "+std::to_string(v);
          for(auto i:ins) { what+=" | in"; for(auto v:tensors[i].shape) what+=" "+std::to_string(v); }
          checked(dml->CreateOperator(&d,IID_PPV_ARGS(&raw)),what.c_str(),device.Get()); }
        if(graph_mode) { nodes.push_back({std::move(raw),ins,out,name}); return; }
        checked(dml->CompileOperator(raw.Get(),DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION,IID_PPV_ARGS(&o.compiled)),"Compile DirectML operator",device.Get());
        setup_operator(o,ins,out);
        ops.push_back(std::move(o));
    }
    void setup_operator(Operator& o,const std::vector<size_t>& ins,size_t out) {
        IDMLCompiledOperator* compiled=o.compiled.Get(); checked(dml->CreateOperatorInitializer(1,&compiled,IID_PPV_ARGS(&o.initializer)),"Create DirectML initializer",device.Get());
        auto props=binding_properties(o.compiled.Get()), init_props=binding_properties(o.initializer.Get());
        check_dml("Get DirectML binding properties");
        const UINT exec_count=graph_mode ? std::max(props.RequiredDescriptorCount,init_props.RequiredDescriptorCount):props.RequiredDescriptorCount;
        const UINT init_count=graph_mode ? exec_count:init_props.RequiredDescriptorCount;
        if(graph_mode) {
            auto show=[](const char* stage,const DML_BINDING_PROPERTIES& p) {
                note(std::string("DirectML graph ")+stage+": descriptors="+std::to_string(p.RequiredDescriptorCount)+" temporary="+std::to_string(p.TemporaryResourceSize)+" persistent="+std::to_string(p.PersistentResourceSize));
            };
            show("execute",props); show("initialize",init_props);
            note("DirectML graph descriptor ranges: execute="+std::to_string(std::max(exec_count,1u))+" initialize="+std::to_string(std::max(init_count,1u))+" (separate heaps)");
        }
        o.descriptors=heap(device.Get(),exec_count); o.init_descriptors=heap(device.Get(),init_count);
        DML_BINDING_TABLE_DESC table{o.compiled.Get(),o.descriptors->GetCPUDescriptorHandleForHeapStart(),o.descriptors->GetGPUDescriptorHandleForHeapStart(),std::max(exec_count,1u)};
        checked(dml->CreateBindingTable(&table,IID_PPV_ARGS(&o.binding)),"Create DirectML binding table",device.Get());
        table={o.initializer.Get(),o.init_descriptors->GetCPUDescriptorHandleForHeapStart(),o.init_descriptors->GetGPUDescriptorHandleForHeapStart(),std::max(init_count,1u)};
        checked(dml->CreateBindingTable(&table,IID_PPV_ARGS(&o.init_binding)),"Create initializer binding table",device.Get());
        if(props.PersistentResourceSize) {
            o.persistent=buffer(device.Get(),props.PersistentResourceSize,D3D12_HEAP_TYPE_DEFAULT);
            bind_resource(o.binding.Get(),o.persistent.Get(),props.PersistentResourceSize,true);
            check_dml("DirectML execute BindPersistentResource");
            auto b=buffer_binding(o.persistent.Get(),props.PersistentResourceSize); DML_BINDING_DESC bd{DML_BINDING_TYPE_BUFFER,&b}; o.init_binding->BindOutputs(1,&bd);
        } else { DML_BINDING_DESC none{DML_BINDING_TYPE_NONE,nullptr}; o.init_binding->BindOutputs(1,&none); }
        check_dml("DirectML initialize BindOutputs (persistent resource)");
        if(props.TemporaryResourceSize) { o.temp=buffer(device.Get(),props.TemporaryResourceSize,D3D12_HEAP_TYPE_DEFAULT); bind_resource(o.binding.Get(),o.temp.Get(),props.TemporaryResourceSize,false); check_dml("DirectML execute BindTemporaryResource"); }
        if(init_props.TemporaryResourceSize) { o.init_temp=buffer(device.Get(),init_props.TemporaryResourceSize,D3D12_HEAP_TYPE_DEFAULT); bind_resource(o.init_binding.Get(),o.init_temp.Get(),init_props.TemporaryResourceSize,false); check_dml("DirectML initialize BindTemporaryResource"); }
        std::vector<DML_BUFFER_BINDING> bindings; std::vector<DML_BINDING_DESC> descs;
        std::vector<DML_BUFFER_BINDING> owned_bindings;
        for(auto id:ins) {
            const auto& t=tensors[id]; auto b=buffer_binding(t.resource.Get(),tensor_bytes(t.shape));
            bindings.push_back(t.owned ? DML_BUFFER_BINDING{nullptr,0,0}:b);
            owned_bindings.push_back(t.owned ? b:DML_BUFFER_BINDING{nullptr,0,0});
            if(graph_mode) {
                std::string name=id==input ? "packed_features":"";
                for(auto& kv:parameters) if(kv.second==id) name=kv.first;
                note("DirectML graph input "+std::to_string(bindings.size()-1)+" "+name+": initialize="+(t.owned ? "BUFFER":"null")+" execute="+(t.owned ? "NONE":"BUFFER")+" tensor_bytes="+std::to_string(tensor_bytes(t.shape))+" buffer_bytes="+std::to_string(b.SizeInBytes));
            }
        }
        for(auto& b:bindings) descs.push_back({b.Buffer ? DML_BINDING_TYPE_BUFFER:DML_BINDING_TYPE_NONE,b.Buffer ? &b:nullptr});
        // One compiled operator, with one entry per input, including null non-owned inputs.
        DML_BUFFER_ARRAY_BINDING array{UINT(owned_bindings.size()),owned_bindings.data()};
        DML_BINDING_DESC init{DML_BINDING_TYPE_BUFFER_ARRAY,&array}; o.init_binding->BindInputs(1,&init);
        check_dml("DirectML initialize BindInputs (one BUFFER_ARRAY)");
        o.binding->BindInputs(static_cast<UINT>(descs.size()),descs.data());
        check_dml("DirectML execute BindInputs");
        auto ob=buffer_binding(tensors[out].resource.Get(),tensor_bytes(tensors[out].shape)); DML_BINDING_DESC od{DML_BINDING_TYPE_BUFFER,&ob}; o.binding->BindOutputs(1,&od);
        check_dml("DirectML execute BindOutputs");
    }
    void compile_graph() {
        std::vector<size_t> ins{input,head_weight,head_bias};
        for(auto& kv:parameters) if(kv.second!=head_weight && kv.second!=head_bias) ins.push_back(kv.second);
        std::map<size_t,UINT> inputs,producer;
        for(UINT i=0;i<ins.size();++i) if(!inputs.emplace(ins[i],i).second) throw std::runtime_error("duplicate DirectML graph input");
        std::set<IDMLOperator*> unique_operators;
        std::vector<DML_OPERATOR_GRAPH_NODE_DESC> operators;
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> input_edges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediate_edges;
        for(UINT i=0;i<nodes.size();++i) {
            if(!unique_operators.insert(nodes[i].raw.Get()).second) throw std::runtime_error("reused DirectML graph operator");
            operators.push_back({nodes[i].raw.Get(),nodes[i].name.c_str()});
            for(UINT j=0;j<nodes[i].inputs.size();++j) {
                size_t id=nodes[i].inputs[j]; auto found=inputs.find(id);
                if(found!=inputs.end()) input_edges.push_back({found->second,i,j,nullptr});
                else {
                    auto source=producer.find(id);
                    if(source==producer.end() || source->second>=i) throw std::runtime_error("missing or forward DirectML graph producer");
                    intermediate_edges.push_back({source->second,0,i,j,nullptr});
                }
            }
            if(inputs.count(nodes[i].output) || !producer.emplace(nodes[i].output,i).second) throw std::runtime_error("duplicate DirectML graph producer");
        }
        std::vector<DML_GRAPH_NODE_DESC> graph_nodes;
        std::vector<DML_GRAPH_EDGE_DESC> ie,me;
        for(auto& n:operators) graph_nodes.push_back({DML_GRAPH_NODE_TYPE_OPERATOR,&n});
        for(auto& e:input_edges) ie.push_back({DML_GRAPH_EDGE_TYPE_INPUT,&e});
        for(auto& e:intermediate_edges) me.push_back({DML_GRAPH_EDGE_TYPE_INTERMEDIATE,&e});
        DML_OUTPUT_GRAPH_EDGE_DESC out{producer.at(output),0,0,nullptr};
        DML_GRAPH_EDGE_DESC oe{DML_GRAPH_EDGE_TYPE_OUTPUT,&out};
        DML_GRAPH_DESC desc{UINT(ins.size()),1,UINT(graph_nodes.size()),graph_nodes.data(),UINT(ie.size()),ie.data(),1,&oe,UINT(me.size()),me.data()};
        // Bindings stay fixed for the graph's lifetime; volatile descriptors would inhibit driver optimisations.
        note("DirectML graph: inputs="+std::to_string(ins.size())+" nodes="+std::to_string(nodes.size())+" input_edges="+std::to_string(ie.size())+" intermediate_edges="+std::to_string(me.size())+" outputs=1");
        checked(dml1->CompileGraph(&desc,DML_EXECUTION_FLAG_ALLOW_HALF_PRECISION_COMPUTATION,IID_PPV_ARGS(&graph.compiled)),"Compile DirectML graph",device.Get());
        check_dml("DirectML graph after CompileGraph");
        // Use default-heap UAV buffers for owned inputs as well as runtime inputs.
        for(auto id:ins) tensors[id].resource=buffer(device.Get(),tensor_bytes(tensors[id].shape),D3D12_HEAP_TYPE_DEFAULT);
        tensors[output].resource=buffer(device.Get(),tensor_bytes(tensors[output].shape),D3D12_HEAP_TYPE_DEFAULT);
        setup_operator(graph,ins,output);
        nodes.clear();
        note(weights.config.kpn() ? "DirectML KPN trunk: compiled graph (all weights owned)":"DirectML trunk: compiled graph (static weights owned, FiLM head runtime inputs)");
    }
    size_t conv(size_t in,const std::string& name,int stride,bool relu) {
        size_t wi=parameter(name+".weight"),bi=parameter(name+".bias",true); auto s=tensors[in].shape;
        Shape target{1,tensors[wi].shape[0],(s[2]+stride-1)/stride,(s[3]+stride-1)/stride}; size_t out=tensor(target);
        TensorDesc a(s),b(tensors[wi].shape,tensors[wi].owned),c(tensors[bi].shape,tensors[bi].owned),z(target);
        UINT strides[]={UINT(stride),UINT(stride)},dilations[]={1,1},pads[]={b.sizes[2]/2,b.sizes[3]/2},zero[]={0,0};
        DML_ACTIVATION_RELU_OPERATOR_DESC act{}; DML_OPERATOR_DESC activation{DML_OPERATOR_ACTIVATION_RELU,&act};
        DML_CONVOLUTION_OPERATOR_DESC d{&a.desc,&b.desc,&c.desc,&z.desc,DML_CONVOLUTION_MODE_CROSS_CORRELATION,DML_CONVOLUTION_DIRECTION_FORWARD,2,strides,dilations,pads,pads,zero,1,relu?&activation:nullptr};
        op(DML_OPERATOR_CONVOLUTION,&d,{in,wi,bi},out); return out;
    }
    size_t add(size_t a,size_t b) {
        auto s=tensors[a].shape; if(s!=tensors[b].shape) throw std::runtime_error("skip shape mismatch"); size_t out=tensor(s);
        TensorDesc x(s),y(s),z(s); DML_ELEMENT_WISE_ADD_OPERATOR_DESC d{&x.desc,&y.desc,&z.desc}; op(DML_OPERATOR_ELEMENT_WISE_ADD,&d,{a,b},out); return out;
    }
    size_t shuffle(size_t a) {
        auto s=tensors[a].shape; Shape t{1,s[1]/4,s[2]*2,s[3]*2}; size_t out=tensor(t);
        TensorDesc x(s),y(t); DML_DEPTH_TO_SPACE1_OPERATOR_DESC d{&x.desc,&y.desc,2,d2s_alt?DML_DEPTH_SPACE_ORDER_DEPTH_COLUMN_ROW:DML_DEPTH_SPACE_ORDER_COLUMN_ROW_DEPTH};
        op(DML_OPERATOR_DEPTH_TO_SPACE1,&d,{a},out); return out;
    }
    size_t pool(size_t a) {
        auto s=tensors[a].shape; Shape t{1,s[1],s[2]/2,s[3]/2}; size_t out=tensor(t);
        TensorDesc x(s),y(t); UINT stride[]={2,2},window[]={2,2},pad[]={0,0};
        DML_MAX_POOLING_OPERATOR_DESC d{&x.desc,&y.desc,2,stride,window,pad,pad};
        op(DML_OPERATOR_MAX_POOLING,&d,{a},out); return out;
    }
    size_t resample(size_t a) {
        auto s=tensors[a].shape; Shape t{1,s[1],s[2]*2,s[3]*2}; size_t out=tensor(t);
        TensorDesc x(s),y(t);
        // output = (input + .5) * scale - .5; torch align_corners=False.
        float scales[]={1,1,2,2},in_offsets[]={.5f,.5f,.5f,.5f},out_offsets[]={-.5f,-.5f,-.5f,-.5f};
        DML_RESAMPLE1_OPERATOR_DESC d{&x.desc,&y.desc,DML_INTERPOLATION_MODE_LINEAR,4,scales,in_offsets,out_offsets};
        op(DML_OPERATOR_RESAMPLE1,&d,{a},out); return out;
    }
    void kpn_trunk() {
        UINT stride=weights.config.trunk_stride;
        input=tensor({1,UINT(weights.config.input_channels()),hp/stride,wp/stride}); size_t x=input; std::vector<size_t> skips;
        for(int i=0;i<5;++i) {
            if(i) x=pool(x);
            x=conv(x,"trunk.enc."+std::to_string(i),1,true); skips.push_back(x);
        }
        x=conv(x,"trunk.bottleneck",1,true);
        for(int i=0;i<4;++i) {
            size_t skip=skips[3-i];
            if(tensors[skip].shape[1]!=tensors[x].shape[1]) skip=conv(skip,"trunk.skip."+std::to_string(i),1,false);
            x=conv(add(resample(x),skip),"trunk.dec."+std::to_string(i)+".0",1,true);
            if(!weights.config.lite) x=conv(x,"trunk.dec."+std::to_string(i)+".2",1,true);
        }
        output=conv(x,"trunk.head",1,false);
        head_weight=parameters.at("trunk.head.weight"); head_bias=parameters.at("trunk.head.bias");
    }
    Impl(ID3D12Device* dev,const Weights& model,UINT width,UINT height,const PipelineOptions& opt):device(dev),weights(model),math(weights),w(width),h(height) {
        const UINT ring_frames=opt.ring_frames; d2s_alt=opt.depth_to_space_alt; fast_path=opt.fast_path;
        if(ring_frames<2 || ring_frames>64) throw std::runtime_error("ring_frames must be 2..64");
        if(!w || !h || w>8192 || h>8192) throw std::runtime_error("render dimension limit");
        UINT multiple=weights.config.kpn() ? 16*weights.config.trunk_stride:1u<<weights.config.widths.size(); wp=(w+multiple-1)/multiple*multiple; hp=(h+multiple-1)/multiple*multiple;
        HMODULE module=load_directml();
        if(!module) throw std::runtime_error("DirectML.dll not found beside the proxy or in System32");
        using Create=HRESULT(WINAPI*)(ID3D12Device*,DML_CREATE_DEVICE_FLAGS,REFIID,void**);
        auto create=reinterpret_cast<Create>(GetProcAddress(module,"DMLCreateDevice"));
        if(!create) throw std::runtime_error("DMLCreateDevice unavailable");
        auto flags=opt.debug_layer ? DML_CREATE_DEVICE_FLAG_DEBUG:DML_CREATE_DEVICE_FLAG_NONE;
        HRESULT hr=create(device.Get(),flags,IID_PPV_ARGS(&dml));
        if(FAILED(hr) && opt.debug_layer) {
            char message[160]; std::snprintf(message,sizeof(message),"DMLCreateDevice(DEBUG) failed: 0x%08X; GetDeviceRemovedReason()=0x%08X; retrying without DirectML debug",unsigned(hr),unsigned(device->GetDeviceRemovedReason()));
            note(message); drain_debug_messages(device.Get()); dml.Reset();
            hr=create(device.Get(),DML_CREATE_DEVICE_FLAG_NONE,IID_PPV_ARGS(&dml));
            flags=DML_CREATE_DEVICE_FLAG_NONE;
        }
        checked(hr,"DMLCreateDevice",device.Get());
        note(flags==DML_CREATE_DEVICE_FLAG_DEBUG ? "DirectML device debug: enabled":"DirectML device debug: disabled");
        checked(dml->CreateCommandRecorder(IID_PPV_ARGS(&recorder)),"Create DirectML command recorder",device.Get());
        graph_mode=(weights.config.kpn() || opt.dml_graph) && SUCCEEDED(dml->QueryInterface(IID_PPV_ARGS(&dml1)));
        if(weights.config.kpn() && !graph_mode) throw std::runtime_error("KPN requires DirectML compiled graph support (IDMLDevice1)");
        if(!graph_mode) note(opt.dml_graph ? "DirectML trunk: per-operator fallback (IDMLDevice1 unavailable)":"DirectML trunk: per-operator (--no-dml-graph)");
        if(weights.config.kpn()) kpn_trunk();
        else {
            input=tensor({1,UINT(weights.config.input_channels()),hp/2,wp/2}); size_t x=conv(input,"stem",1,true); std::vector<size_t> skips;
            for(size_t i=0;i<weights.config.widths.size();++i) {
                for(int j=0;j<weights.config.depths[i];++j) { auto prefix="enc."+std::to_string(i)+"."+std::to_string(j); x=add(x,conv(conv(x,prefix+".a",1,true),prefix+".b",1,false)); }
                if(i+1<weights.config.widths.size()) { skips.push_back(x); x=conv(x,"down."+std::to_string(i),2,true); }
            }
            for(size_t i=skips.size();i-->0;) x=add(shuffle(conv(x,"up."+std::to_string(i),1,false)),skips[i]);
            output=conv(conv(x,"fuse",1,true),"out",1,false); head_weight=parameters.at("out.weight"); head_bias=parameters.at("out.bias");
        }
        for(auto& kv:initial_data) {
            auto r=upload(device.Get(),kv.second.data(),kv.second.size());
            buffer_binding(r.Get(),tensor_bytes(tensors[kv.first].shape));
            initial_uploads[kv.first]=std::move(r);
        }
        if(graph_mode) compile_graph();
        bias_offset=constant_buffer_bytes+(weights.config.kpn() ? 0:tensors[head_weight].resource->GetDesc().Width);
        slots.resize(ring_frames);
        for(auto& slot:slots) {
            slot.descriptors=heap(device.Get(),56);
            slot.data=buffer(device.Get(),bias_offset+(weights.config.kpn() ? 0:tensors[head_bias].resource->GetDesc().Width),D3D12_HEAP_TYPE_UPLOAD);
            D3D12_RANGE empty{0,0}; checked(slot.data->Map(0,&empty,reinterpret_cast<void**>(&slot.mapped)),"Map ring upload",device.Get());
            std::memset(slot.mapped,0,static_cast<size_t>(slot.data->GetDesc().Width));
        }
        hcl=weights.config.kpn() ? texture(device.Get(),2*w,2*h,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                                  : buffer(device.Get(),uint64_t(w)*h*12*2,D3D12_HEAP_TYPE_DEFAULT);
        // KPN leaves the fast-only descriptor slots backed by small valid buffers.
        hraw=buffer(device.Get(),weights.config.kpn() ? 4:uint64_t(w)*h*12*2,D3D12_HEAP_TYPE_DEFAULT);
        cw=buffer(device.Get(),weights.config.kpn() ? 4:uint64_t(w)*h*4*2,D3D12_HEAP_TYPE_DEFAULT);
        reset_buffer=buffer(device.Get(),weights.config.kpn() ? 4:uint64_t(w)*h*2,D3D12_HEAP_TYPE_DEFAULT);
        const auto history_format=DXGI_FORMAT_R16G16B16A16_FLOAT;
        const auto depth_format=weights.config.kpn() ? DXGI_FORMAT_R32_FLOAT:DXGI_FORMAT_R16_FLOAT;
        for(int i=0;i<2;i++) { history[i]=texture(device.Get(),w*2,h*2,history_format,i==0?D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:D3D12_RESOURCE_STATE_UNORDERED_ACCESS); depth[i]=texture(device.Get(),w,h,depth_format,i==0?D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:D3D12_RESOURCE_STATE_UNORDERED_ACCESS); }
        if(weights.config.kpn()) for(int i=0;i<2;i++)
            tracker[i]=texture(device.Get(),w,h,DXGI_FORMAT_R16G16B16A16_FLOAT,
                i==0 ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if(weights.config.coverage) for(int i=0;i<2;i++)
            coverage_state[i]=texture(device.Get(),w,h,DXGI_FORMAT_R32G32B32A32_FLOAT,
                i==0 ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if(weights.config.history_age) for(int i=0;i<2;i++)
            age_state[i]=texture(device.Get(),w,h,DXGI_FORMAT_R32G32_FLOAT,
                i==0 ? D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto_exposure=texture(device.Get(),1,1,DXGI_FORMAT_R32_FLOAT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        display=texture(device.Get(),w*2,h*2,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        D3D12_DESCRIPTOR_RANGE ranges[4]={{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,7,0,0,0},{D3D12_DESCRIPTOR_RANGE_TYPE_UAV,6,0,0,0},{D3D12_DESCRIPTOR_RANGE_TYPE_UAV,2,6,0,0},{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,7,0,0}};
        D3D12_ROOT_PARAMETER params[6]{}; params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV; params[0].Descriptor={0,0};
        for(int i=0;i<2;i++) { params[i+1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; params[i+1].DescriptorTable={1,&ranges[i]}; }
        params[3].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS; params[3].Constants={1,0,2};
        params[4].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; params[4].DescriptorTable={1,&ranges[2]};
        params[5].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE; params[5].DescriptorTable={1,&ranges[3]};
        D3D12_STATIC_SAMPLER_DESC sampler{};
        sampler.Filter=D3D12_FILTER_MIN_MAG_LINEAR_MIP_POINT;
        sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MaxAnisotropy=1; sampler.ComparisonFunc=D3D12_COMPARISON_FUNC_NEVER;
        sampler.MaxLOD=D3D12_FLOAT32_MAX; sampler.ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC rs{6,params,1,&sampler,D3D12_ROOT_SIGNATURE_FLAG_NONE}; ComPtr<ID3DBlob> blob,errors;
        checked(D3D12SerializeRootSignature(&rs,D3D_ROOT_SIGNATURE_VERSION_1,blob.GetAddressOf(),errors.GetAddressOf()),"Serialize root signature",device.Get());
        checked(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)),"Create root signature",device.Get());
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{}; pso.pRootSignature=root.Get(); pso.CS={pack_dxil,sizeof(pack_dxil)};
        checked(device->CreateComputePipelineState(&pso,IID_PPV_ARGS(&pack)),"Create pack PSO",device.Get()); pso.CS={resolve_dxil,sizeof(resolve_dxil)};
        checked(device->CreateComputePipelineState(&pso,IID_PPV_ARGS(&resolve)),"Create resolve PSO",device.Get()); pso.CS={pack_fast_dxil,sizeof(pack_fast_dxil)};
        checked(device->CreateComputePipelineState(&pso,IID_PPV_ARGS(&pack_fast)),"Create FAST pack PSO",device.Get()); pso.CS={resolve_fast_dxil,sizeof(resolve_fast_dxil)};
        checked(device->CreateComputePipelineState(&pso,IID_PPV_ARGS(&resolve_fast)),"Create FAST resolve PSO",device.Get()); pso.CS={kpn_pack_dxil,sizeof(kpn_pack_dxil)};
        checked(device->CreateComputePipelineState(&pso,IID_PPV_ARGS(&kpn_pack)),"Create KPN pack PSO",device.Get()); pso.CS={kpn_resolve_dxil,sizeof(kpn_resolve_dxil)};
        checked(device->CreateComputePipelineState(&pso,IID_PPV_ARGS(&kpn_resolve)),"Create KPN resolve PSO",device.Get()); pso.CS={kpn_resolve3_dxil,sizeof(kpn_resolve3_dxil)};
        checked(device->CreateComputePipelineState(&pso,IID_PPV_ARGS(&kpn_resolve3)),"Create KPN 3x3 resolve PSO",device.Get()); pso.CS={exposure_dxil,sizeof(exposure_dxil)};
        checked(device->CreateComputePipelineState(&pso,IID_PPV_ARGS(&exposure)),"Create exposure PSO",device.Get()); pso.CS={rcas_dxil,sizeof(rcas_dxil)};
        checked(device->CreateComputePipelineState(&pso,IID_PPV_ARGS(&rcas)),"Create RCAS PSO",device.Get());
    }
};
Pipeline::Pipeline(ID3D12Device* d,const Weights& w,uint32_t width,uint32_t height,const PipelineOptions& o):impl(std::make_unique<Impl>(d,w,width,height,o)) {}
Pipeline::~Pipeline()=default;
void Pipeline::reset() { impl->first=true; }
void Pipeline::record(ID3D12GraphicsCommandList* list,const FrameInputs& in,ID3D12QueryHeap* times) {
    auto& p=*impl; const auto& cfg=p.weights.config;
    if(!list || list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT) throw std::runtime_error("direct command list required");
    if(!in.color.resource || !in.motion.resource || !in.depth.resource || !in.output.resource || !(in.pre_exposure>0)) throw std::runtime_error("invalid frame inputs");
    if(in.sharpen && (!std::isfinite(in.sharpness) || in.sharpness<0 || in.sharpness>1)) throw std::runtime_error("sharpness must be in [0,1]");
    const bool use_auto=!in.model_space && !in.exposure.resource && (in.auto_exposure || !in.exposure_one);
    InputResource exposure=use_auto ? InputResource{p.auto_exposure.Get(),DXGI_FORMAT_R32_FLOAT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS} : in.exposure;
    for(auto r:{in.color,in.motion,in.depth,in.output}) {
        auto desc=r.resource->GetDesc();
        if(desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D || desc.SampleDesc.Count!=1 || desc.DepthOrArraySize!=1 || desc.MipLevels!=1 || r.format==DXGI_FORMAT_UNKNOWN) throw std::runtime_error("unsupported texture layout");
        ComPtr<ID3D12Device> owner; checked(r.resource->GetDevice(IID_PPV_ARGS(&owner)),"Get resource device",p.device.Get());
        if(owner.Get()!=p.device.Get()) throw std::runtime_error("resource device mismatch");
    }
    for(auto r:{in.color,in.motion,in.depth,in.exposure}) if(r.resource==in.output.resource) throw std::runtime_error("output aliases input");
    auto od=in.output.resource->GetDesc();
    if(od.Width<2*p.w || od.Height<2*p.h || !(od.Flags&D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)) throw std::runtime_error("invalid output extent or UAV flags");
    const FrameMath& f=p.math.compute(in.jitter_x,in.jitter_y); Constants c{};
    const bool kpn=cfg.kpn();
    bool fast=!kpn && p.fast_path && cfg.depth_test && cfg.nearest_sample && cfg.conf_consistent && cfg.base_gate && cfg.carry_raw && cfg.bicubic && !cfg.conf_motion && !cfg.depth_dilate && !cfg.thin_lock && !in.model_space && !in.debug_view;
    for(int q=0;q<4;q++) for(int i=0;i<2;i++) fast=fast && f.offsets[2*q+i]>=-1 && f.offsets[2*q+i]<=1 && f.windows[2*q+i]>=0 && f.windows[2*q+i]<=1;
    if(!p.frame) note(kpn ? "Shaders: KPN":fast ? "Shaders: FAST":"Shaders: GENERIC");
    c.w=p.w;c.h=p.h;c.wp=p.wp;c.hp=p.hp;c.first_frame=p.first||in.reset;c.depth_test=cfg.depth_test;c.bicubic=cfg.bicubic;c.carry_raw=cfg.carry_raw;
    c.stabilize=kpn && !c.first_frame && p.stabilizing ? in.stabilize:0;
    c.stabilize_tau=in.stabilize_tau;c.stabilize_eps=in.stabilize_eps;
    const bool stabilized=c.stabilize>0;
    const bool tracking=kpn && in.foliage.enabled();
    c.foliage_strength=kpn ? in.foliage.foliage_strength:0;
    c.fallback_strength=kpn ? in.foliage.fallback_strength:0;
    c.foliage_ema=in.foliage.foliage_ema;c.foliage_threshold=in.foliage.foliage_threshold;
    c.foliage_eps=in.foliage.foliage_eps;c.foliage_spatial=in.foliage.foliage_spatial;
    c.foliage_history_scale=in.foliage.foliage_history_scale;c.foliage_alpha_floor=in.foliage.foliage_alpha_floor;
    c.fallback_sigma=in.foliage.fallback_sigma;c.fallback_alpha=in.foliage.fallback_alpha;
    c.tracker_reset=c.first_frame || (in.foliage.foliage_strength>0)!=(p.last_foliage.foliage_strength>0)
        || (in.foliage.fallback_strength>0)!=(p.last_foliage.fallback_strength>0)
        || in.pre_exposure!=p.last_pre_exposure || in.model_space!=p.last_model_space || in.exposure_one!=p.last_exposure_one;
    c.nearest_sample=cfg.nearest_sample;c.conf_consistent=cfg.conf_consistent;c.base_gate=cfg.base_gate;c.conf_motion=cfg.conf_motion;
    c.robustness=cfg.mv_dilate | (cfg.depth_dilate<<1) | (cfg.thin_lock<<2) | (cfg.depth_soft<<3) | (cfg.coverage<<4) | (cfg.depth_soft_osc<<5) | (cfg.history_age<<6); c.thin_factor=f.thin_factor; c.coverage_floor=f.coverage_floor;
    c.soft_osc_threshold=f.soft_osc_threshold; c.alpha_min=f.alpha_min;
    c.box_slack=kpn ? 0:p.weights.scalar("box_slack");c.conf_max=cfg.conf_max;c.conf_m=cfg.conf_motion?p.weights.scalar("conf_m"):0;c.pre_exposure=in.pre_exposure;
    c.model_space=in.model_space;c.inverted_depth=in.inverted_depth;c.display_motion=in.display_motion;c.exposure_one=use_auto ? false : in.exposure_one;c.debug_view=in.debug_view;c.has_exposure=exposure.resource!=nullptr;
    std::copy(f.signed_jitter.begin(),f.signed_jitter.end(),c.signed_jitter); c.kpn_residual=cfg.residual;
    if(kpn) {
        auto dd=in.depth.resource->GetDesc();
        c.kpn_depth_hr=dd.Width==2*p.w && dd.Height==2*p.h;
        if(!c.kpn_depth_hr && (dd.Width!=p.w || dd.Height!=p.h)) throw std::runtime_error("KPN depth must be render or output size");
        auto md=in.motion.resource->GetDesc();
        if(md.Width!=(in.display_motion ? 2*p.w:p.w) || md.Height!=(in.display_motion ? 2*p.h:p.h)) throw std::runtime_error("KPN motion extent mismatch");
        auto cd=in.color.resource->GetDesc();
        if(cd.Width<p.w || cd.Height<p.h) throw std::runtime_error("KPN color extent mismatch");
    }
    std::copy(in.motion_scale,in.motion_scale+2,c.motion_scale); std::copy(in.jitter_cancel,in.jitter_cancel+2,c.jitter_cancel);
    std::copy(f.phase.begin(),f.phase.end(),c.phase); std::copy(f.kernels.begin(),f.kernels.end(),c.kernels);
    // KPN has no phase weights; reuse their constant slots for the resolve options.
    if(kpn) { c.robustness|=(cfg.trunk_stride==2 ? 128u:0u)|(p.d2s_alt ? 256u:0u)|(cfg.history_filter=="catmull" ? 512u:0u); c.phase[0]=cfg.sigma_min; c.phase[1]=cfg.proximity; c.phase[2]=cfg.proximity_gain; }
    for(int q=0;q<4;q++) for(int i=0;i<2;i++) { c.offsets[q][i]=f.offsets[2*q+i];c.windows[q][i]=f.windows[2*q+i]; }
    auto& batch=p.slots[p.frame%p.slots.size()];
    std::memcpy(batch.mapped,&c,sizeof(c));
    if(!kpn) {
        std::memcpy(batch.mapped+constant_buffer_bytes,f.head_weight.data(),f.head_weight.size()*2);
        std::memcpy(batch.mapped+p.bias_offset,f.head_bias.data(),f.head_bias.size()*2);
    }
    const auto constants=batch.data->GetGPUVirtualAddress();
    UINT step=p.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu=[&](UINT i){auto h=batch.descriptors->GetCPUDescriptorHandleForHeapStart();h.ptr+=SIZE_T(i)*step;return h;};
    auto gpu=[&](UINT i){auto h=batch.descriptors->GetGPUDescriptorHandleForHeapStart();h.ptr+=UINT64(i)*step;return h;};
    auto srv=[&](UINT i,ID3D12Resource* r,DXGI_FORMAT format,bool is_buffer=false) {
        if(format==DXGI_FORMAT_UNKNOWN) format=DXGI_FORMAT_R32_FLOAT;
        D3D12_SHADER_RESOURCE_VIEW_DESC d{}; d.Format=format;d.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        if(is_buffer) { d.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;d.Buffer.NumElements=UINT(r->GetDesc().Width/(format==DXGI_FORMAT_R32_FLOAT ? 4:2)); }
        else { d.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;d.Texture2D.MipLevels=1; }
        p.device->CreateShaderResourceView(r,&d,cpu(i));
    };
    auto uav=[&](UINT i,ID3D12Resource* r,DXGI_FORMAT format,bool is_buffer=false) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};d.Format=format;
        if(is_buffer) {d.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;d.Buffer.NumElements=UINT(r->GetDesc().Width/(format==DXGI_FORMAT_R32_FLOAT ? 4:2));} else d.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
        p.device->CreateUnorderedAccessView(r,nullptr,&d,cpu(i));
    };
    const int prev=p.ping,next=1-prev;
    const auto history_format=DXGI_FORMAT_R16G16B16A16_FLOAT;
    const auto depth_format=kpn ? DXGI_FORMAT_R32_FLOAT:DXGI_FORMAT_R16_FLOAT;
    const auto history_buffer_format=kpn ? DXGI_FORMAT_R16G16B16A16_FLOAT:DXGI_FORMAT_R16_FLOAT;
    srv(0,p.history[prev].Get(),history_format);srv(1,in.color.resource,in.color.format);srv(2,in.motion.resource,in.motion.format);srv(3,in.depth.resource,in.depth.format);srv(4,p.depth[prev].Get(),depth_format);srv(5,exposure.resource,exposure.format);srv(6,p.coverage_state[prev].Get(),DXGI_FORMAT_R32G32B32A32_FLOAT);
    uav(7,p.tensors[p.input].resource.Get(),DXGI_FORMAT_R16_FLOAT,true);uav(8,p.hcl.Get(),history_buffer_format,!kpn);uav(9,p.hraw.Get(),DXGI_FORMAT_R16_FLOAT,true);uav(10,p.cw.Get(),DXGI_FORMAT_R16_FLOAT,true);uav(11,p.reset_buffer.Get(),DXGI_FORMAT_R16_FLOAT,true);uav(12,p.depth[next].Get(),depth_format);
    srv(13,p.tensors[p.output].resource.Get(),DXGI_FORMAT_R16_FLOAT,true);srv(14,in.color.resource,in.color.format);srv(15,p.hcl.Get(),history_buffer_format,!kpn);srv(16,(stabilized || tracking) ? p.tensors[p.input].resource.Get():p.hraw.Get(),DXGI_FORMAT_R16_FLOAT,true);srv(17,p.cw.Get(),DXGI_FORMAT_R16_FLOAT,true);srv(18,p.reset_buffer.Get(),DXGI_FORMAT_R16_FLOAT,true);srv(19,exposure.resource,exposure.format);
    if(kpn) {
        srv(6,p.tracker[prev].Get(),history_format);
        uav(9,p.tracker[next].Get(),history_format);
        srv(17,p.tracker[next].Get(),history_format);
        srv(18,p.depth[next].Get(),depth_format);
    }
    uav(20,p.history[next].Get(),history_format);uav(21,in.sharpen ? p.display.Get() : in.output.resource,in.sharpen ? DXGI_FORMAT_R16G16B16A16_FLOAT : in.output.format);
    uav(52,p.coverage_state[next].Get(),DXGI_FORMAT_R32G32B32A32_FLOAT);
    uav(53,p.age_state[next].Get(),DXGI_FORMAT_R32G32_FLOAT);
    srv(54,p.age_state[prev].Get(),DXGI_FORMAT_R32G32_FLOAT);
    srv(55,p.age_state[next].Get(),DXGI_FORMAT_R32G32_FLOAT);
    for(UINT i=22;i<26;i++) uav(i,nullptr,DXGI_FORMAT_R16G16B16A16_FLOAT);
    // Additional passes each occupy seven SRVs followed by six UAVs in every ring slot.
    srv(26,in.color.resource,in.color.format);
    srv(39,p.display.Get(),DXGI_FORMAT_R16G16B16A16_FLOAT);srv(40,exposure.resource,exposure.format);
    for(UINT i=27;i<33;i++) srv(i,nullptr,DXGI_FORMAT_R32_FLOAT);
    for(UINT i=41;i<46;i++) srv(i,nullptr,DXGI_FORMAT_R32_FLOAT);
    uav(33,p.auto_exposure.Get(),DXGI_FORMAT_R32_FLOAT);uav(46,in.output.resource,in.output.format);
    for(UINT i=34;i<39;i++) uav(i,nullptr,DXGI_FORMAT_R32_FLOAT);
    for(UINT i=47;i<52;i++) uav(i,nullptr,DXGI_FORMAT_R32_FLOAT);
    uint32_t pass_constants[2]={uint32_t(c.first_frame || !p.exposure_ready),0};
    const float rcas_linear=std::exp2(-2*(1-in.sharpness));
    std::memcpy(&pass_constants[1],&rcas_linear,sizeof(rcas_linear));
    // All frame allocations and input validation are above the first command-list mutation.
    auto copy=[&](size_t id,ID3D12Resource* source,uint64_t offset) {
        auto dst=p.tensors[id].resource.Get();
        transition(list,dst,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(dst,0,source,offset,dst->GetDesc().Width);
        transition(list,dst,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    };
    if(!p.initialized) for(auto& kv:p.initial_uploads) copy(kv.first,kv.second.Get(),0);
    if(!kpn) { copy(p.head_weight,batch.data.Get(),constant_buffer_bytes); copy(p.head_bias,batch.data.Get(),p.bias_offset); }
    if(!p.initialized && p.graph_mode) {
        ID3D12DescriptorHeap* hs[]={p.graph.init_descriptors.Get()};list->SetDescriptorHeaps(1,hs);
        p.recorder->RecordDispatch(list,p.graph.initializer.Get(),p.graph.init_binding.Get());uav_barrier(list);
        p.check_dml("DirectML graph initialize RecordDispatch");
    }
    if(!p.initialized) for(auto& op:p.ops) { ID3D12DescriptorHeap* hs[]={op.init_descriptors.Get()};list->SetDescriptorHeaps(1,hs);p.recorder->RecordDispatch(list,op.initializer.Get(),op.init_binding.Get());uav_barrier(list); }
    for(auto r:{in.color,in.motion,in.depth,in.exposure}) if(r.resource) transition(list,r.resource,r.state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(list,in.output.resource,in.output.state,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto shader=[&](ID3D12PipelineState* state,UINT start) {
        ID3D12DescriptorHeap* hs[]={batch.descriptors.Get()};list->SetDescriptorHeaps(1,hs);list->SetComputeRootSignature(p.root.Get());list->SetPipelineState(state);list->SetComputeRootConstantBufferView(0,constants);list->SetComputeRootDescriptorTable(1,gpu(start));list->SetComputeRootDescriptorTable(2,gpu(start+7));list->SetComputeRoot32BitConstants(3,2,pass_constants,0);list->SetComputeRootDescriptorTable(4,gpu(52));list->SetComputeRootDescriptorTable(5,gpu(start==13 ? 55:54));
    };
    if(times) list->EndQuery(times,D3D12_QUERY_TYPE_TIMESTAMP,0);
    if(use_auto) {
        shader(p.exposure.Get(),26);list->Dispatch(1,1,1);uav_barrier(list);
        transition(list,p.auto_exposure.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    shader(kpn ? p.kpn_pack.Get():fast ? p.pack_fast.Get():p.pack.Get(),0);list->Dispatch((p.wp+7)/8,(p.hp+7)/8,1);uav_barrier(list);
    if(times) list->EndQuery(times,D3D12_QUERY_TYPE_TIMESTAMP,1);
    if(p.graph_mode) {
        ID3D12DescriptorHeap* hs[]={p.graph.descriptors.Get()};list->SetDescriptorHeaps(1,hs);
        p.recorder->RecordDispatch(list,p.graph.compiled.Get(),p.graph.binding.Get());uav_barrier(list);
        p.check_dml("DirectML graph execute RecordDispatch");
    }
    for(auto& op:p.ops) {ID3D12DescriptorHeap* hs[]={op.descriptors.Get()};list->SetDescriptorHeaps(1,hs);p.recorder->RecordDispatch(list,op.compiled.Get(),op.binding.Get());uav_barrier(list);}
    if(times) list->EndQuery(times,D3D12_QUERY_TYPE_TIMESTAMP,2);
    if(stabilized || tracking) transition(list,p.tensors[p.input].resource.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    for(auto r:{p.tensors[p.output].resource.Get(),p.hcl.Get(),p.hraw.Get(),p.cw.Get(),p.reset_buffer.Get()}) transition(list,r,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if(kpn) {
        transition(list,p.tracker[next].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(list,p.depth[next].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    if(cfg.history_age) transition(list,p.age_state[next].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    shader(kpn ? (cfg.taps==3 ? p.kpn_resolve3.Get():p.kpn_resolve.Get()):fast ? p.resolve_fast.Get():p.resolve.Get(),13);list->Dispatch((p.w+7)/8,(p.h+7)/8,1);uav_barrier(list);
    if(in.sharpen) {
        transition(list,p.display.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        shader(p.rcas.Get(),39);list->Dispatch((2*p.w+7)/8,(2*p.h+7)/8,1);uav_barrier(list);
        transition(list,p.display.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    if(use_auto) transition(list,p.auto_exposure.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if(times) list->EndQuery(times,D3D12_QUERY_TYPE_TIMESTAMP,3);
    if(stabilized || tracking) transition(list,p.tensors[p.input].resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    for(auto r:{p.tensors[p.output].resource.Get(),p.hcl.Get(),p.hraw.Get(),p.cw.Get(),p.reset_buffer.Get()}) transition(list,r,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    for(auto r:{in.color,in.motion,in.depth,in.exposure}) if(r.resource) transition(list,r.resource,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,r.state);
    transition(list,in.output.resource,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,in.output.state);
    for(auto r:{p.history[prev].Get(),p.depth[prev].Get()}) transition(list,r,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(list,p.history[next].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if(kpn) transition(list,p.tracker[prev].Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    else transition(list,p.depth[next].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    p.last_foliage=in.foliage;p.last_pre_exposure=in.pre_exposure;
    p.last_model_space=in.model_space;p.last_exposure_one=in.exposure_one;
    if(cfg.history_age) transition(list,p.age_state[prev].Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if(cfg.coverage) {
        transition(list,p.coverage_state[prev].Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(list,p.coverage_state[next].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    p.exposure_ready=use_auto;p.ping=next;p.first=false;p.initialized=true;p.stabilizing=in.stabilize>0;++p.frame;
}
}
