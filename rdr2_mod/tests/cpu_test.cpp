#include "cpu_math.h"
#include "runtime_config.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace fsrmamba;
namespace {
void check(bool ok,const char* text) { if(!ok) throw std::runtime_error(text); }
template<class T> T read(std::istream& f) { T t{}; f.read(reinterpret_cast<char*>(&t),sizeof(t)); check(bool(f),"truncated reference"); return t; }
void near(float a,float b,float eps) { if(!std::isfinite(a)||!std::isfinite(b)||std::abs(a-b)>eps) { std::cerr<<a<<" != "<<b<<'\n'; throw std::runtime_error("reference mismatch"); } }
}
int main(int argc,char** argv) {
    try {
        check(argc==2 || argc==3,"usage: cpu_test weights.bin [reference.bin]");
        for(const wchar_t* text:{L"",L" ",L"bad",L".6junk",L"nan",L"inf",L"-inf",L"1e1000"})
            near(runtime_float(text,.5f,.05f,4),.5f,0);
        near(runtime_float(L" .6 \t",0,0,.95f),.6f,0);
        near(runtime_float(L"-1",0,0,.95f),0,0);
        near(runtime_float(L"2",0,0,.95f),.95f,0);
        near(runtime_float(L"0",.5f,.05f,4),.05f,0);
        near(runtime_float(L"5",.5f,.05f,4),4,0);
        near(runtime_float(L"0",.004f,1e-4f,.1f),1e-4f,0);
        near(runtime_float(L".2",.004f,1e-4f,.1f),.1f,0);
        near(runtime_float(L".004",.004f,1e-4f,.1f),.004f,0);
        for(const auto& setting:foliage_settings) {
            for(const wchar_t* text:{L"",L"bad",L"nan",L"inf",L".3junk"}) near(foliage_float(text,setting),setting.start,0);
            near(foliage_float(L"0",setting),setting.minimum>0 ? setting.start:0,0);
            near(foliage_float(L"-5",setting),setting.minimum,0);
            near(foliage_float(L"10",setting),setting.maximum,0);
        }
        check(runtime_key(L"VK_END")==0x23 && runtime_key(L"VK_OEM_4")==0xdb,"named keys");
        check(runtime_key(L"0xDB")==0xdb && runtime_key(L"219")==0xdb,"numeric keys");
        check(runtime_key(L"VK_F12")==0x7b && runtime_key(L"VK_NUMPAD3")==0x63,"function/numpad keys");
        for(const wchar_t* key:{L"",L"bad",L"VK_UNKNOWN",L"0",L"256",L"-1",L"219junk",L"VK_F25"})
            check(runtime_key(key)==0,"invalid key accepted");
        LivePreset preset;
        for(const wchar_t* text:{L"",L" \t\n"}) {
            check(parse_preset(text,preset)==PresetParse::empty && !preset.valid,"empty preset");
        }
        for(const wchar_t* text:{L"bad=1",L"enable=1",L"stabilize",L"stabilize=",L"stabilize=bad",
                L"stabilize=nan",L"stabilize=inf",L"stabilize=1e1000",L"stabilize=.6junk",
                L"stabilize=0,",L",stabilize=0",L"stabilize=0,,foliage_strength=1",
                L"stabilize=0,bad=1",L"stabilize=0,foliage_strength=.5=1"}) {
            check(parse_preset(L"stabilize=.85",preset)==PresetParse::valid,"preset setup");
            check(parse_preset(text,preset)==PresetParse::invalid && !preset.valid,"bad preset accepted");
            LiveControls controls; preset.apply(controls);
            near(controls.stabilize,0,0);
        }
        for(size_t i=0;i<live_setting_count;++i) for(const wchar_t* value:{L"0",L"-5",L"10",L".25"}) {
            check(parse_preset(std::wstring(live_setting_name(i))+L"="+value,preset)==PresetParse::valid,"valid live setting");
            LiveControls expected,actual;
            apply_live_setting(expected,i,value); preset.apply(actual);
            near(live_setting_value(actual,i),live_setting_value(expected,i),0);
        }
        check(parse_preset(L" stabilize = 2 , foliage_strength = 1 , fallback_strength = .5, foliage_ema=0 ",preset)==PresetParse::valid,"valid preset list");
        LiveControls controls; preset.apply(controls);
        near(controls.stabilize,.95f,0); near(controls.foliage.foliage_strength,1,0);
        near(controls.foliage.fallback_strength,.5f,0); near(controls.foliage.foliage_ema,.2f,0);
        check(parse_preset(L"stabilize=.85,stabilize=0",preset)==PresetParse::valid,"duplicate setting");
        preset.apply(controls); near(controls.stabilize,0,0);
        PresetCycle cycle;
        check(cycle.advance()==0 && cycle.active==-1,"empty cycle");
        parse_preset(L"stabilize=0,foliage_strength=0,fallback_strength=0",cycle.presets[0]);
        parse_preset(L"bad=1",cycle.presets[1]);
        parse_preset(L"foliage_strength=1,fallback_strength=.5",cycle.presets[2]);
        parse_preset(L"stabilize=.85,foliage_strength=0",cycle.presets[5]);
        LiveControls ini; ini.stabilize=.4f; ini.foliage.foliage_history_scale=.6f;
        check(cycle.advance()==1,"first preset");
        near(cycle.apply(ini).stabilize,0,0);
        check(cycle.advance()==3,"skip invalid/missing presets");
        near(cycle.apply(ini).stabilize,.4f,0);
        near(cycle.apply(ini).foliage.fallback_strength,.5f,0);
        near(cycle.apply(ini).foliage.foliage_history_scale,.6f,0);
        cycle.ini_reload(false);
        check(cycle.active==2,"unchanged ini lost preset");
        near(cycle.apply(ini).foliage.foliage_strength,1,0);
        check(cycle.advance()==6 && cycle.advance()==1,"preset wrap-around");
        cycle.ini_reload(true); ini.stabilize=.2f; ini.foliage.foliage_strength=.3f;
        check(cycle.active==-1,"changed ini did not reset preset");
        near(cycle.apply(ini).stabilize,.2f,0); near(cycle.apply(ini).foliage.foliage_strength,.3f,0);
        check(cycle.advance()==1,"cycle restart after ini change");
        PresetCycle single; parse_preset(L"stabilize=0",single.presets[4]);
        check(single.advance()==5 && single.advance()==5,"single preset wrap");
        auto w=Weights::read(argv[1]);
        if(argc==2) {
            for(auto& kv:w.tensors) {
                double sum=0, weighted=0; auto values=kv.second.floats();
                for(size_t i=0;i<values.size();++i) { sum+=values[i]; weighted+=values[i]*double(i+1); }
                std::cout<<kv.first<<' '<<values.size()<<' '<<std::setprecision(17)<<sum<<' '<<weighted<<'\n';
            }
            return 0;
        }
        for(uint32_t h=0;h<65536;++h) if((h&0x7c00)!=0x7c00) check(to_half(from_half(uint16_t(h)))==h,"half roundtrip");
        for(int n=-8;n<=8;++n) check(round_even(n+.5)==(n%2==0?n:n+1),"round to even");
        near(exposure_scale(2,3,true),1.5f,0); near(exposure_scale(0,0,false),1,0);
        std::ifstream f(argv[2],std::ios::binary); check(bool(f),"cannot open reference");
        if(w.config.kpn()) {
            for(bool flag:{w.config.lite,w.config.residual,w.config.mv_dilate,w.config.depth_soft})
                check(flag==(read<uint32_t>(f)!=0),"KPN flag mismatch");
            near(w.config.jitter_sign,read<float>(f),0);
            for(auto* dims:{&w.config.widths,&w.config.render_size,&w.config.output_size,&w.config.padded_size})
                for(int d:*dims) check(d==int(read<uint32_t>(f)),"KPN dimension mismatch");
            for(float value:{w.config.sigma_min,w.config.proximity,w.config.proximity_gain}) near(value,read<float>(f),0);
            check(w.config.trunk_stride==int(read<uint32_t>(f)),"KPN stride");
            check(w.config.taps==int(read<uint32_t>(f)),"KPN taps");
            check((w.config.history_filter=="catmull")==bool(read<uint32_t>(f)),"KPN history filter");
            const int phases=w.config.trunk_stride*w.config.trunk_stride;
            check(w.config.input_channels()==16*phases && w.config.output_channels()==28*phases,"KPN channels");
            FrameMaths math(w);
            for(int i=0;i<3;++i) {
                double x=read<double>(f),y=read<double>(f); const auto& frame=math.compute(x,y);
                near(frame.signed_jitter[0],read<float>(f),0); near(frame.signed_jitter[1],read<float>(f),0);
                check(frame.head_weight.empty() && frame.head_bias.empty(),"KPN head must be static");
            }
            const auto images=read<uint32_t>(f);
            for(uint32_t i=0;i<images;++i) {
                const auto h=read<uint32_t>(f),width=read<uint32_t>(f),n=4*h*width;
                const double x=read<double>(f),y=read<double>(f);
                const auto jitter=math.compute(x,y).signed_jitter;
                std::vector<float> rgb(3*h*width),history(3*n),params(7*n),reset(n),alpha(n),out(3*n);
                for(auto* array:{&rgb,&history,&params,&reset,&alpha,&out}) for(auto& v:*array) v=read<float>(f);
                for(uint32_t oy=0;oy<2*h;++oy) for(uint32_t ox=0;ox<2*width;++ox) {
                    const auto at=oy*2*width+ox;
                    const auto value=kpn_pixel(w.config,rgb.data(),history.data()+3*at,params.data()+7*at,
                                               jitter,h,width,oy,ox,reset[at]>.5f);
                    near(value.alpha,alpha[at],2e-6f);
                    for(int c=0;c<3;++c) near(value.color[c],out[3*at+c],2e-5f);
                }
                for(int mode=0;mode<4;++mode) {
                    FoliageControls controls; controls.foliage_strength=float(mode&1); controls.fallback_strength=float(mode>>1);
                    std::vector<float> evidence(3*n),expected(3*n);
                    for(auto* array:{&evidence,&expected}) for(auto& v:*array) v=read<float>(f);
                    for(uint32_t oy=0;oy<2*h;++oy) for(uint32_t ox=0;ox<2*width;++ox) {
                        const auto at=oy*2*width+ox;
                        auto value=kpn_pixel(w.config,rgb.data(),history.data()+3*at,params.data()+7*at,
                                             jitter,h,width,oy,ox,reset[at]>.5f,controls,
                                             {evidence[3*at],evidence[3*at+1],evidence[3*at+2]});
                        for(int c=0;c<3;++c) near(value.color[c],expected[3*at+c],2e-5f);
                        if(mode==0) {
                            const auto base=kpn_pixel(w.config,rgb.data(),history.data()+3*at,params.data()+7*at,
                                                       jitter,h,width,oy,ox,reset[at]>.5f);
                            for(int c=0;c<3;++c) near(value.color[c],base.color[c],0);
                        }
                    }
                }
            }
            const auto trackers=read<uint32_t>(f);
            for(uint32_t i=0;i<trackers;++i) {
                float values[15]; for(auto& v:values) v=read<float>(f);
                auto result=foliage_tracker(values[0],values[1],values[2],{values[3],values[4],values[5],values[6]},
                                            values[7],values[8],values[9]>.5f,FoliageControls{});
                for(int c=0;c<4;++c) near(result.state[c],values[10+c],.00025f);
                near(result.confidence,values[14],2e-6f);
            }
            check(f.peek()==std::char_traits<char>::eof(),"trailing KPN reference bytes");
            std::cout<<"PASS: KPN flags, sizes, options, static head, signed jitter and "<<images<<" resolve images (off/A/D/combined), "<<trackers<<" tracker cases\n";
            return 0;
        }
        check(w.config.depth_soft==(read<uint32_t>(f)!=0),"depth_soft configuration mismatch");
        check(w.config.coverage==(read<uint32_t>(f)!=0),"coverage configuration mismatch");
        near(w.config.coverage_bias,read<float>(f),0);
        near(frame_math(w,0,0).coverage_floor,read<float>(f),0);
        const auto count=read<uint32_t>(f);
        for(uint32_t i=0;i<count;++i) {
            const auto x=read<double>(f),y=read<double>(f); auto v=frame_math(w,x,y);
            for(auto a:v.signed_jitter) near(a,read<float>(f),0);
            for(auto a:v.phase) near(a,read<float>(f),0);
            for(auto a:v.offsets) check(a==read<int32_t>(f),"sample offset mismatch");
            for(auto a:v.windows) check(a==read<int32_t>(f),"window mismatch");
            for(auto a:v.kernels) near(a,read<float>(f),2e-6f);
            for(auto* arr:{&v.head_weight,&v.head_bias}) {
                check(arr->size()==read<uint32_t>(f),"head length mismatch");
                for(auto a:*arr) { float b=read<float>(f); near(from_half(a),b,std::max(2e-6f,std::abs(b)*.0011f)); }
            }
        }
        near(frame_math(w,0,0).thin_factor,read<float>(f),0);
        const auto images=read<uint32_t>(f);
        for(uint32_t image=0;image<images;++image) {
            const auto h=read<uint32_t>(f),width=read<uint32_t>(f),n=h*width;
            std::vector<float> rgb(3*n),motion(2*n),depth(n);
            for(auto* array:{&rgb,&motion,&depth}) for(auto& v:*array) v=read<float>(f);
            for(uint32_t y=0;y<h;++y) for(uint32_t x=0;x<width;++x) {
                auto value=robust_pixel(w,rgb.data(),motion.data(),depth.data(),h,width,y,x);
                for(auto v:value.motion) near(v,read<float>(f),0);
                near(value.depth,read<float>(f),0);
                near(value.thin,read<float>(f),0);
                for(auto v:value.slack) near(v,read<float>(f),0);
            }
        }
        const auto coefficients=read<uint32_t>(f);
        for(uint32_t i=0;i<coefficients;++i) {
            float logit=read<float>(f); near(coverage_coefficient(logit,w.config.coverage_bias),read<float>(f),0);
        }
        const auto coverage_frames=read<uint32_t>(f);
        for(uint32_t i=0;i<coverage_frames;++i) {
            const auto h=read<uint32_t>(f),width=read<uint32_t>(f),n=h*width;
            std::vector<float> rgb(3*n),previous(4*n),motion(2*n),reset(n),evidence(2*n),next(4*n);
            for(auto* array:{&rgb,&previous,&motion,&reset,&evidence,&next}) for(auto& v:*array) v=read<float>(f);
            for(uint32_t y=0;y<h;++y) for(uint32_t x=0;x<width;++x) {
                uint32_t at=y*width+x;
                auto value=coverage_pixel(rgb.data(),previous.data(),{motion[2*at],motion[2*at+1]},h,width,y,x,reset[at]>.5f);
                for(int c=0;c<2;++c) near(value.evidence[c],evidence[2*at+c],.002f);
                for(int c=0;c<4;++c) near(value.state[c],next[4*at+c],2e-7f);
            }
        }
        check(w.config.depth_soft_osc==(read<uint32_t>(f)!=0),"depth_soft_osc configuration mismatch");
        near(frame_math(w,0,0).soft_osc_threshold,read<float>(f),0);
        const auto depth_cases=read<uint32_t>(f);
        for(uint32_t i=0;i<depth_cases;++i) {
            bool offscreen=read<uint32_t>(f)!=0,first=read<uint32_t>(f)!=0;
            float prev=read<float>(f),mn=read<float>(f),mx=read<float>(f),osc=read<float>(f);
            bool expected=read<uint32_t>(f)!=0;
            check(depth_reset(w,offscreen,first,prev,mn,mx,osc)==expected,"depth reset mismatch");
        }
        check(w.config.history_age==(read<uint32_t>(f)!=0),"history_age configuration mismatch");
        float floor=read<float>(f); near(frame_math(w,0,0).alpha_min,floor,0);
        const auto age_images=read<uint32_t>(f);
        for(uint32_t i=0;i<age_images;++i) {
            const auto h=read<uint32_t>(f),width=read<uint32_t>(f),n=h*width;
            std::vector<float> previous(n),motion(2*n),reset(n),alpha(n),expected(4*n);
            for(auto* array:{&previous,&motion,&reset,&alpha,&expected}) for(auto& v:*array) v=read<float>(f);
            for(uint32_t y=0;y<h;++y) for(uint32_t x=0;x<width;++x) {
                uint32_t at=y*width+x;
                auto value=age_pixel(previous.data(),{motion[2*at],motion[2*at+1]},h,width,y,x,reset[at]>.5f);
                near(value.age,expected[4*at],0); near(value.next,expected[4*at+1],0);
                near(value.feature,expected[4*at+2],0);
                float u=(x+.5f)/width+motion[2*at],v=(y+.5f)/h+motion[2*at+1];
                bool hard=reset[at]>.5f || u<0 || u>=1 || v<0 || v>=1;
                near(age_alpha(alpha[at],value.age,floor,hard),expected[4*at+3],0);
            }
        }
        check(f.peek()==std::char_traits<char>::eof(),"trailing reference bytes");
        std::cout<<"PASS: "<<count<<" jitter cases; phase, offsets, kernels, FiLM head, half conversion, exposure; "<<images<<" robustness images; "<<coverage_frames<<" coverage frames; "<<depth_cases<<" depth reset cases; "<<age_images<<" age images\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
