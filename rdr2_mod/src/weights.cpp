#include "weights.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

namespace fsrmamba {
namespace {
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Json {
    enum Kind { number, boolean, string, array, object } kind = object;
    double n = 0;
    std::string s;
    std::vector<Json> a;
    std::map<std::string, Json> o;
    const Json& at(const std::string& key) const {
        auto i = o.find(key); require(i != o.end(), "missing configuration key"); return i->second;
    }
    bool flag() const { require(kind == boolean, "expected boolean"); return n != 0; }
    double num() const { require(kind == number, "expected number"); return n; }
    std::string str() const { require(kind == string, "expected string"); return s; }
};
struct JsonReader {
    const std::string& s; size_t p = 0;
    void ws() { while (p < s.size() && (s[p]==' ' || s[p]=='\n' || s[p]=='\r' || s[p]=='\t')) ++p; }
    char get() { require(p < s.size(), "truncated JSON"); return s[p++]; }
    void expect(char c) { ws(); require(get() == c, "invalid JSON delimiter"); }
    std::string quoted() {
        expect('"'); std::string out;
        for (;;) {
            char c = get(); if (c == '"') return out;
            require(static_cast<unsigned char>(c) >= 32 && static_cast<unsigned char>(c) < 127 && c != '\\', "configuration strings must be plain ASCII");
            out += c;
        }
    }
    Json value(int depth = 0) {
        require(depth < 8, "JSON nesting limit"); ws(); require(p < s.size(), "truncated JSON"); Json j;
        if (s[p] == '{') {
            ++p; ws(); if (p < s.size() && s[p] == '}') { ++p; return j; }
            for (;;) { auto key = quoted(); expect(':'); auto v = value(depth+1);
                require(j.o.emplace(key, std::move(v)).second, "duplicate configuration key");
                ws(); char c = get(); if (c == '}') return j; require(c == ',', "invalid JSON object"); }
        }
        if (s[p] == '[') {
            j.kind = Json::array; ++p; ws(); if (p < s.size() && s[p] == ']') { ++p; return j; }
            for (;;) { require(j.a.size() < 64, "configuration array limit"); j.a.push_back(value(depth+1));
                ws(); char c = get(); if (c == ']') return j; require(c == ',', "invalid JSON array"); }
        }
        if (s[p] == '"') { j.kind = Json::string; j.s = quoted(); return j; }
        for (auto lit : {"true", "false"}) if (s.compare(p, std::strlen(lit), lit) == 0) {
            p += std::strlen(lit); j.kind = Json::boolean; j.n = lit[0] == 't'; return j;
        }
        const size_t start = p;
        if (s[p] == '-') ++p;
        require(p < s.size() && s[p] >= '0' && s[p] <= '9', "invalid JSON number");
        if (s[p] == '0') ++p; else while (p < s.size() && s[p]>='0' && s[p]<='9') ++p;
        if (p < s.size() && s[p] == '.') { ++p; const auto b=p; while (p<s.size() && s[p]>='0' && s[p]<='9') ++p; require(p>b,"invalid JSON fraction"); }
        if (p < s.size() && (s[p]=='e' || s[p]=='E')) { ++p; if (p<s.size() && (s[p]=='+' || s[p]=='-')) ++p;
            const auto b=p; while (p<s.size() && s[p]>='0' && s[p]<='9') ++p; require(p>b,"invalid JSON exponent"); }
        j.kind=Json::number; j.n=std::stod(s.substr(start,p-start)); require(std::isfinite(j.n),"nonfinite configuration"); return j;
    }
};
Config parse_config(const std::string& text) {
    JsonReader r{text}; auto j=r.value(); r.ws(); require(r.p==text.size(),"trailing JSON");
    if(j.at("arch").str()=="kpn") {
        const std::set<std::string> keys={"arch","scale","widths","lite","residual","mv_dilate",
            "depth_soft","jitter_sign","preset","render_size","output_size","padded_size"};
        const std::set<std::string> optional={"sigma_min","proximity","proximity_gain","trunk_stride","taps","history_filter"};
        require(j.kind==Json::object,"invalid KPN configuration fields");
        for(auto& key:keys) j.at(key);
        for(auto& kv:j.o) require(keys.count(kv.first)!=0 || optional.count(kv.first)!=0,"unknown KPN configuration key");
        Config c; c.arch="kpn"; c.history_age=true; c.depth_test=true; c.bicubic=true;
        auto option=[&](const char* name,double fallback) { return j.o.count(name) ? j.at(name).num():fallback; };
        const double stride=option("trunk_stride",1),taps=option("taps",5);
        require(stride==1 || stride==2,"invalid KPN trunk stride");
        require(taps==3 || taps==5,"invalid KPN taps");
        c.trunk_stride=int(stride); c.taps=int(taps);
        if(j.o.count("history_filter")) c.history_filter=j.at("history_filter").str();
        require(c.history_filter=="bicubic" || c.history_filter=="catmull","invalid KPN history filter");
        const double sigma_min=option("sigma_min",.3),proximity=option("proximity",0),gain=option("proximity_gain",2);
        require(std::isfinite(sigma_min) && sigma_min>=1e-3 && sigma_min<=2.5,
                "KPN sigma_min must be finite and in [1e-3, 2.5]");
        require(std::isfinite(proximity) && (proximity==0 || (proximity>=1e-3 && proximity<=16)),
                "KPN proximity must be 0 (off) or finite and in [1e-3, 16]");
        require(std::isfinite(gain) && gain>=1e-3 && gain<=64,
                "KPN proximity_gain must be finite and in [1e-3, 64]");
        c.sigma_min=float(sigma_min); c.proximity=float(proximity); c.proximity_gain=float(gain);
        require(j.at("scale").num()==2,"KPN requires 2x scale");
        c.lite=j.at("lite").flag(); c.residual=j.at("residual").flag();
        c.mv_dilate=j.at("mv_dilate").flag(); c.depth_soft=j.at("depth_soft").flag();
        double sign=j.at("jitter_sign").num(); require(sign==1 || sign==-1,"invalid KPN jitter sign"); c.jitter_sign=float(sign);
        auto dims=[&](const char* name,size_t count,int limit) {
            const auto& a=j.at(name); require(a.kind==Json::array && a.a.size()==count,"invalid KPN dimensions");
            std::vector<int> out;
            for(auto& v:a.a) { double n=v.num(); require(n==std::floor(n) && n>=1 && n<=limit,"invalid KPN dimension"); out.push_back(int(n)); }
            return out;
        };
        c.widths=dims("widths",6,16384); c.render_size=dims("render_size",2,8192);
        c.output_size=dims("output_size",2,16384); c.padded_size=dims("padded_size",2,8192);
        const int multiple=16*c.trunk_stride;
        for(int i=0;i<2;++i) require(c.output_size[i]==2*c.render_size[i] && c.padded_size[i]==(c.render_size[i]+multiple-1)/multiple*multiple,"invalid KPN sizes");
        c.preset=j.at("preset").str();
        require(c.preset=="custom" || (c.preset=="lite" && c.lite && c.widths==std::vector<int>{16,24,48,64,96,128}) ||
            (c.preset=="default" && !c.lite && c.widths==std::vector<int>{24,32,64,96,128,192}),"invalid KPN preset");
        return c;
    }
    const std::set<std::string> keys = {"arch","scale","n_state","stem_kernel","learned_clamp","detail_ch","resolve","hist_residual","accum","widths","depths","film","depth_test","nearest_sample","conf_consistent","carry_raw","base_gate","hist_filter","conf_motion","jitter_sign","conf_max"};
    const std::set<std::string> optional = {"history_age","coverage_bias","coverage","depth_soft","depth_soft_osc","mv_dilate","depth_dilate","thin_lock"};
    require(j.kind==Json::object,"unsupported configuration fields");
    for (auto& key:keys) j.at(key);
    for (auto& kv:j.o) require(keys.count(kv.first)!=0 || optional.count(kv.first)!=0,"unknown configuration key");
    require(j.at("arch").str()=="fast" && j.at("scale").num()==2 && j.at("n_state").num()==0 && j.at("stem_kernel").num()==1 && !j.at("learned_clamp").flag() && j.at("detail_ch").num()==0 && j.at("resolve").str()=="nearest" && !j.at("hist_residual").flag() && j.at("accum").flag(),"unsupported model configuration");
    Config c;
    if(j.o.count("coverage_bias")) c.coverage_bias=static_cast<float>(j.at("coverage_bias").num());
    require(std::isfinite(c.coverage_bias) && c.coverage_bias<0,"invalid coverage bias");
    if(j.o.count("history_age")) c.history_age=j.at("history_age").flag();
    if(j.o.count("coverage")) c.coverage=j.at("coverage").flag();
    if(j.o.count("depth_soft")) c.depth_soft=j.at("depth_soft").flag();
    if(j.o.count("depth_soft_osc")) c.depth_soft_osc=j.at("depth_soft_osc").flag();
    if(j.o.count("mv_dilate")) c.mv_dilate=j.at("mv_dilate").flag();
    if(j.o.count("depth_dilate")) c.depth_dilate=j.at("depth_dilate").flag();
    if(j.o.count("thin_lock")) c.thin_lock=j.at("thin_lock").flag();
    for (auto name : {"widths","depths"}) {
        const auto& v=j.at(name); require(v.kind==Json::array && !v.a.empty() && v.a.size()<=12,"invalid pyramid");
        auto& dst=std::string(name)=="widths" ? c.widths:c.depths;
        for (auto& x:v.a) { double n=x.num(); require(n==std::floor(n) && n>= (std::string(name)=="widths"?1:0) && n<=16384,"invalid pyramid dimension"); dst.push_back(static_cast<int>(n)); }
    }
    require(c.widths.size()==c.depths.size(),"pyramid length mismatch");
    c.film=j.at("film").flag(); c.depth_test=j.at("depth_test").flag(); c.nearest_sample=j.at("nearest_sample").flag();
    require(!c.depth_soft || c.depth_test,"depth_soft requires depth_test");
    require(!c.depth_soft_osc || (c.depth_test && c.depth_soft && c.coverage),"depth_soft_osc requires depth_test, depth_soft and coverage");
    c.conf_consistent=j.at("conf_consistent").flag(); c.carry_raw=j.at("carry_raw").flag(); c.base_gate=j.at("base_gate").flag();
    c.conf_motion=j.at("conf_motion").flag(); auto filter=j.at("hist_filter").str(); require(filter=="bilinear" || filter=="bicubic","invalid history filter"); c.bicubic=filter=="bicubic";
    c.jitter_sign=static_cast<float>(j.at("jitter_sign").num()); c.conf_max=static_cast<float>(j.at("conf_max").num());
    require((c.jitter_sign==1 || c.jitter_sign==-1) && c.conf_max>0 && c.conf_max<=65504,"invalid scalar configuration"); return c;
}
struct Reader {
    std::ifstream f; uint64_t remaining;
    explicit Reader(const std::filesystem::path& path):f(path,std::ios::binary),remaining(0) {
        require(bool(f),"cannot open weights"); f.seekg(0,std::ios::end); auto end=f.tellg();
        require(end>=0 && end <= (1LL<<30),"weights file size limit"); remaining=static_cast<uint64_t>(end); f.seekg(0);
    }
    void read(void* p,size_t n) { require(n<=remaining,"truncated weights"); f.read(static_cast<char*>(p),static_cast<std::streamsize>(n)); require(bool(f),"weights read failed"); remaining-=n; }
    uint32_t u32() { uint8_t b[4]; read(b,4); return b[0] | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24; }
    uint64_t u64() { auto lo=u32(); return lo | uint64_t(u32())<<32; }
    std::string str(size_t n) { require(n<=65536,"string limit"); std::string s(n,'\0'); read(s.data(),n); return s; }
};
}
uint16_t to_half(float v) {
    uint32_t u; std::memcpy(&u,&v,4); const uint32_t sign=(u>>16)&0x8000, mant=u&0x7fffff; const int e=int((u>>23)&255)-127;
    if (e==128) return static_cast<uint16_t>(sign|0x7c00|(mant?0x200:0));
    if (e>15) return static_cast<uint16_t>(sign|0x7c00);
    if (e<-25) return static_cast<uint16_t>(sign);
    const int shift=e<-14 ? -e-1 : 13;
    const uint32_t m=e<-14 ? mant|0x800000 : mant;
    uint32_t q=m>>shift, rem=m&((1u<<shift)-1), mid=1u<<(shift-1);
    q += rem>mid || (rem==mid && (q&1));
    return static_cast<uint16_t>(sign + (e<-14 ? 0 : uint32_t(e+15)<<10) + q);
}
float from_half(uint16_t h) {
    float v;
    const int e=(h>>10)&31, m=h&1023;
    if (e==31) v=m ? std::numeric_limits<float>::quiet_NaN():std::numeric_limits<float>::infinity();
    else v=std::ldexp(float(e ? 1024+m:m),e ? e-25:-24);
    return h&0x8000 ? -v:v;
}
std::vector<float> Tensor::floats() const {
    std::vector<float> out; out.reserve(bytes.size()/(dtype==1?2:4));
    for (size_t i=0;i<bytes.size();) {
        const uint32_t lo=bytes[i]|uint32_t(bytes[i+1])<<8;
        if (dtype==1) { out.push_back(from_half(static_cast<uint16_t>(lo))); i+=2; }
        else { uint32_t u=lo|uint32_t(bytes[i+2])<<16|uint32_t(bytes[i+3])<<24; float v; std::memcpy(&v,&u,4); out.push_back(v); i+=4; }
    }
    return out;
}
const Tensor& Weights::at(const std::string& name) const { auto i=tensors.find(name); require(i!=tensors.end(),"missing tensor"); return i->second; }
float Weights::scalar(const std::string& name) const { const auto& t=at(name); require(t.shape.empty(),"expected scalar"); return t.floats()[0]; }
Weights Weights::read(const std::filesystem::path& path) {
    Reader r(path); require(r.str(8)==std::string("FSMWGT1\0",8),"invalid weights magic"); Weights w; w.config=parse_config(r.str(r.u32()));
    const auto count=r.u32(); require(count>0 && count<=10000,"tensor count limit");
    for (uint32_t i=0;i<count;++i) {
        auto name=r.str(r.u32()); require(!name.empty() && name.size()<=255,"invalid tensor name");
        require(std::all_of(name.begin(),name.end(),[](char c){return (c>='a'&&c<='z') || (c>='0'&&c<='9') || c=='_' || c=='.';}),"invalid tensor name");
        Tensor t; t.dtype=r.u32(); require(t.dtype==1 || t.dtype==2,"unsupported tensor dtype"); auto rank=r.u32(); require(rank<=4,"tensor rank limit");
        uint64_t elements=1;
        for (uint32_t d=0;d<rank;++d) { auto dim=r.u32(); require(dim>0 && dim<=16384 && elements<=(1u<<28)/dim,"tensor dimension limit"); elements*=dim; t.shape.push_back(dim); }
        uint64_t bytes=r.u64(); require(bytes==elements*(t.dtype==1?2:4) && bytes<=r.remaining,"invalid tensor byte count"); t.bytes.resize(static_cast<size_t>(bytes)); r.read(t.bytes.data(),t.bytes.size());
        for(float v:t.floats()) require(std::isfinite(v),"nonfinite tensor value");
        require(w.tensors.emplace(name,std::move(t)).second,"duplicate tensor");
    }
    require(r.remaining==0,"trailing weights data"); w.validate(); return w;
}
void Weights::validate() const {
    std::set<std::string> used;
    auto tensor=[&](std::string name,std::vector<uint32_t> shape,uint32_t dtype) {
        const auto& t=at(name); require(t.shape==shape && t.dtype==dtype,"tensor shape/dtype mismatch"); used.insert(name);
    };
    auto conv=[&](std::string name,int in,int out,int k,uint32_t dtype=1) {
        tensor(name+".weight",{uint32_t(out),uint32_t(in),uint32_t(k),uint32_t(k)},dtype); tensor(name+".bias",{uint32_t(out)},dtype);
    };
    if(config.kpn()) {
        require(config.widths.size()==6,"KPN requires six widths");
        int cin=config.input_channels();
        for(int i=0;i<5;++i) { conv("trunk.enc."+std::to_string(i),cin,config.widths[i],i==1 ? 1:3); cin=config.widths[i]; }
        conv("trunk.bottleneck",cin,config.widths[5],1); cin=config.widths[5];
        for(int i=0;i<4;++i) {
            int cout=config.widths[3-i];
            if(cin!=cout) conv("trunk.skip."+std::to_string(i),cout,cin,1);
            conv("trunk.dec."+std::to_string(i)+".0",cin,cout,1);
            if(!config.lite) conv("trunk.dec."+std::to_string(i)+".2",cout,cout,1);
            cin=cout;
        }
        conv("trunk.head",cin,config.output_channels(),1);
        require(used.size()==tensors.size(),"unexpected KPN tensor"); return;
    }
    conv("stem",config.input_channels(),config.widths[0],1);
    for (size_t i=0;i<config.widths.size();++i) {
        int c=config.widths[i];
        for(int j=0;j<config.depths[i];++j) for(auto suffix:{".a",".b"}) conv("enc."+std::to_string(i)+"."+std::to_string(j)+suffix,c,c,3);
        if(i+1<config.widths.size()) { conv("down."+std::to_string(i),c,config.widths[i+1],3); conv("up."+std::to_string(i),config.widths[i+1],c*4,1); }
    }
    conv("fuse",config.widths[0],config.widths[0],3); conv("out",config.widths[0],config.output_channels(),1,2);
    tensor("box_slack",{},2); tensor("acc_sharp",{},2);
    if(config.history_age) tensor("alpha_min",{},2);
    if(config.conf_motion) tensor("conf_m",{},2);
    if(config.thin_lock) tensor("thin_slack",{},2);
    if(config.depth_soft_osc) tensor("soft_osc_threshold",{},2);
    if(config.film) {
        tensor("film.0.weight",{32,2},2); tensor("film.0.bias",{32},2);
        tensor("film.2.weight",{uint32_t(2*config.widths[0]),32},2); tensor("film.2.bias",{uint32_t(2*config.widths[0])},2);
    }
    require(used.size()==tensors.size(),"unexpected tensor");
}
}
