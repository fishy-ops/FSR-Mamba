#include "cpu_math.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fsrmamba {
int round_even(double x) {
    if (!std::isfinite(x) || std::abs(x) > 16384) throw std::runtime_error("invalid sample offset");
    const double lo = std::floor(x), t = x - lo;
    const int n = static_cast<int>(lo);
    return n + (t > .5 || (t == .5 && n % 2 != 0));
}
static float lanczos(float x) {
    x = std::min(std::abs(x), 2.f);
    if (x < 1e-5f) return 1;
    const float p = 3.14159265358979323846f * x;
    return (std::sin(p) / p) * (std::sin(p * .5f) / (p * .5f));
}
float exposure_scale(float pre, float exposure, bool fsr) {
    if (!fsr) return 1;
    if (!(pre > 0) || !(exposure > 0) || !std::isfinite(pre) || !std::isfinite(exposure))
        throw std::runtime_error("invalid exposure");
    return exposure / pre;
}
FrameMaths::FrameMaths(const Weights& weights):w(&weights) {
    if(weights.config.kpn()) return;
    out_w=weights.at("out.weight").floats(); out_b=weights.at("out.bias").floats();
    if(weights.config.film) {
        film_a=weights.at("film.0.weight").floats(); film_b=weights.at("film.0.bias").floats();
        film_c=weights.at("film.2.weight").floats(); film_d=weights.at("film.2.bias").floats();
        gb.resize(2*weights.config.widths[0]);
    }
    weight.resize(out_w.size()); bias.resize(out_b.size());
    sharp=std::abs(weights.scalar("acc_sharp"));
    result.head_weight.resize(out_w.size()); result.head_bias.resize(out_b.size());
    if(weights.config.history_age) result.alpha_min=std::clamp(weights.scalar("alpha_min"),0.f,1.f);
    result.coverage_floor=coverage_floor(weights.config.coverage_bias);
    if(weights.config.depth_soft_osc)
        result.soft_osc_threshold=std::clamp(weights.scalar("soft_osc_threshold"),.05f,4.f);
    if(weights.config.thin_lock)
        result.thin_factor=from_half(to_half(1+weights.scalar("thin_slack")));
}
const FrameMath& FrameMaths::compute(double x,double y) {
    const Weights& w=*this->w;
    if (!std::isfinite(x) || !std::isfinite(y) || std::abs(x) > .5 || std::abs(y) > .5)
        throw std::runtime_error("jitter outside supported half-pixel range");
    x *= w.config.jitter_sign; y *= w.config.jitter_sign;
    FrameMath& f=result;
    f.signed_jitter = {static_cast<float>(x), static_cast<float>(y)};
    if(w.config.kpn()) return f;
    for (int q = 0; q < 4; ++q) {
        const float px = .25f + .5f * (q % 2), py = .25f + .5f * (q / 2);
        float dx = 1.f - f.signed_jitter[0] - px;
        float dy = 1.f - f.signed_jitter[1] - py;
        dx = dx - std::floor(dx) - .5f; dy = dy - std::floor(dy) - .5f;
        f.phase[q] = from_half(to_half(std::exp(-sharp * (dx * dx + dy * dy))));
        const bool fp32_offsets = w.config.base_gate && !w.config.carry_raw;
        f.offsets[2*q] = w.config.nearest_sample ? round_even(py - .5 + (fp32_offsets ? double(f.signed_jitter[1]) : y)) : 0;
        f.offsets[2*q+1] = w.config.nearest_sample ? round_even(px - .5 + (fp32_offsets ? double(f.signed_jitter[0]) : x)) : 0;
        const int sx = .5f - f.signed_jitter[0] - px > 0 ? -2 : -1;
        const int sy = .5f - f.signed_jitter[1] - py > 0 ? -2 : -1;
        f.windows[2*q] = sy + 2; f.windows[2*q+1] = sx + 2;
        float sum = 0;
        for (int t = 0; t < 16; ++t) {
            const int tx = t % 4 - 2, ty = t / 4 - 2;
            const float ox = tx + .5f - f.signed_jitter[0] - px;
            const float oy = ty + .5f - f.signed_jitter[1] - py;
            const float k = tx >= sx && tx <= sx+2 && ty >= sy && ty <= sy+2 ? lanczos(std::sqrt(ox*ox + oy*oy)) : 0;
            f.kernels[q*16+t] = k; sum += k;
        }
        for (int t = 0; t < 16; ++t) f.kernels[q*16+t] /= std::max(sum, 1e-4f);
    }
    const int channels = w.config.widths[0], outputs = w.config.output_channels();
    std::copy(out_w.begin(),out_w.end(),weight.begin()); std::copy(out_b.begin(),out_b.end(),bias.begin());
    if (w.config.film) {
        float hidden[32];
        for (int i = 0; i < 32; ++i) hidden[i] = std::max(0.f, film_a[2*i]*f.signed_jitter[0] + film_a[2*i+1]*f.signed_jitter[1] + film_b[i]);
        for (int i = 0; i < 2*channels; ++i) {
            float v = 0;
            for (int j = 0; j < 32; ++j) v += film_c[i*32+j]*hidden[j];
            gb[i] = v + film_d[i];
        }
        for (int o = 0; o < outputs; ++o) {
            float v = 0;
            for (int i = 0; i < channels; ++i) {
                v += gb[channels+i]*weight[o*channels+i];
                weight[o*channels+i] *= 1 + gb[i];
            }
            bias[o] += v;
        }
    }
    for (size_t i=0;i<weight.size();++i) f.head_weight[i]=to_half(weight[i]);
    for (size_t i=0;i<bias.size();++i) f.head_bias[i]=to_half(bias[i]);
    return f;
}
FrameMath frame_math(const Weights& w, double x, double y) { FrameMaths m(w); return m.compute(x,y); }
KPNPixel kpn_pixel(const Config& cfg,const float* rgb,const float* history,const float* p,
                   std::array<float,2> jitter,int h,int w,int y,int x,bool reset,
                   const FoliageControls& controls,std::array<float,3> evidence) {
    const float k=controls.foliage_strength>0 ? evidence[0]:0;
    const float confidence=evidence[1],bad=controls.fallback_strength>0 ? evidence[2]:0;
    const float xy[2]={(x+.5f)*.5f-.5f+jitter[0],(y+.5f)*.5f-.5f+jitter[1]};
    const int cx=std::clamp(int(std::floor(xy[0]+.5f)),0,w-1);
    const int cy=std::clamp(int(std::floor(xy[1]+.5f)),0,h-1);
    float a=std::max(std::exp(std::clamp(p[0],std::log(cfg.sigma_min),std::log(2.5f))),1e-4f);
    float b=std::max(std::exp(std::clamp(p[1],std::log(cfg.sigma_min),std::log(2.5f))),1e-4f);
    if(bad>0) { a+=(std::max(a,controls.fallback_sigma)-a)*bad; b+=(std::max(b,controls.fallback_sigma)-b)*bad; }
    const float cs=std::cos(p[2]),sn=std::sin(p[2]);
    const int radius=cfg.taps/2,count=cfg.taps*cfg.taps;
    float logits[25],peak=-INFINITY;
    for(int oy=-radius;oy<=radius;++oy) for(int ox=-radius;ox<=radius;++ox) {
        const float dx=cx+ox-xy[0],dy=cy+oy-xy[1];
        const float u=(cs*dx+sn*dy)/a,v=(-sn*dx+cs*dy)/b;
        const int i=(oy+radius)*cfg.taps+ox+radius;
        logits[i]=cx+ox>=0 && cx+ox<w && cy+oy>=0 && cy+oy<h ? -.5f*(u*u+v*v):-INFINITY;
        peak=std::max(peak,logits[i]);
    }
    float total=0;
    for(int i=0;i<count;++i) { logits[i]=std::exp(logits[i]-peak); total+=logits[i]; }
    auto sample=[&](int dy,int dx,int c) {
        int at=std::clamp(cy+dy,0,h-1)*w+std::clamp(cx+dx,0,w-1);
        return from_half(to_half(rgb[3*at+c]));
    };
    auto sigmoid=[](float v) {
        const float e=std::exp(-std::abs(v));
        return v>=0 ? 1/(1+e):e/(1+e);
    };
    KPNPixel result{};
    result.alpha=sigmoid(p[3]);
    if(cfg.proximity>0) {
        const float dx=(cx-xy[0])*2,dy=(cy-xy[1])*2,r2=dx*dx+dy*dy;
        if(bad>0) result.alpha=sigmoid(p[3]+(std::log(cfg.proximity_gain)-r2/(2*cfg.proximity*cfg.proximity))*(1-bad));
        else result.alpha=sigmoid(p[3]+std::log(cfg.proximity_gain)-r2/(2*cfg.proximity*cfg.proximity));
    }
    if(bad>0) result.alpha+=(std::max(result.alpha,controls.fallback_alpha)-result.alpha)*bad;
    if(k>0) result.alpha+=(std::min(result.alpha,std::max(controls.foliage_alpha_floor,result.alpha*controls.foliage_history_scale))-result.alpha)*k*confidence*(1-bad);
    if(reset) result.alpha=1;
    const float slack=sigmoid(p[4]),residual[3]={p[5]+p[6],p[5],p[5]-p[6]};
    for(int c=0;c<3;++c) {
        float current=0,lo=INFINITY,hi=-INFINITY;
        for(int oy=-radius;oy<=radius;++oy) for(int ox=-radius;ox<=radius;++ox)
            current+=logits[(oy+radius)*cfg.taps+ox+radius]/std::max(total,1e-8f)*sample(oy,ox,c);
        for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
            float value=sample(dy,dx,c); lo=std::min(lo,value); hi=std::max(hi,value);
        }
        if(k>0) {
            float spatial=0,norm=0;
            for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
                float xx=cx+dx-xy[0],yy=cy+dy-xy[1];
                float weight=cx+dx>=0 && cx+dx<w && cy+dy>=0 && cy+dy<h ? std::exp(-(xx*xx+yy*yy)/(2*.65f*.65f)):0;
                spatial+=weight*sample(dy,dx,c); norm+=weight;
            }
            current+=(spatial/std::max(norm,1e-8f)-current)*k*controls.foliage_spatial*(1-bad);
        }
        float rectified=std::clamp(history[c],lo-slack*(hi-lo),hi+slack*(hi-lo));
        float correction=cfg.residual ? .05f*std::tanh(residual[c]):0;
        result.color[c]=bad>0 && reset ? current:std::clamp(result.alpha*current+(1-result.alpha)*rectified+correction,0.f,1-1e-6f);
    }
    return result;
}
FoliageTracker foliage_tracker(float L,float B,float range,const std::array<float,4>& old,
                                float old_B,float spread,bool invalid,const FoliageControls& c) {
    FoliageTracker result{{L,0,0,B},0};
    if(!invalid) {
        float t=std::clamp((std::abs(B-old_B)-.03f)/.03f,0.f,1.f),agreement=1-t*t*(3-2*t);
        float d=(L-B)-(old[0]-old[3]);
        float event=d*old[1]<-c.foliage_eps*c.foliage_eps ? std::clamp(std::min(std::abs(d),std::abs(old[1]))/(range+c.foliage_eps),0.f,1.f):0;
        result.state[1]=agreement>0 ? d:0;
        result.state[2]=(old[2]+(event-old[2])*c.foliage_ema)*agreement;
        result.confidence=agreement*std::clamp(1-spread/1.5f,0.f,1.f);
    }
    for(auto& v:result.state) v=from_half(to_half(v));
    return result;
}
RobustPixel robust_pixel(const Weights& weights,const float* rgb,const float* motion,
                         const float* depth,int h,int w,int y,int x) {
    auto rh=[](float v) { return from_half(to_half(v)); };
    auto at=[&](int dy,int dx) { return std::clamp(y+dy,0,h-1)*w+std::clamp(x+dx,0,w-1); };
    int best=at(-1,-1), centre=at(0,0);
    float near=rh(depth[best]), l[9], lo=3e38f, hi=-3e38f;
    float mn[3]={3e38f,3e38f,3e38f},mx[3]={-3e38f,-3e38f,-3e38f};
    for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        int i=at(dy,dx),q=(dy+1)*3+dx+1;
        float d=rh(depth[i]); if(d>near) { near=d; best=i; }
        float c[3];
        for(int k=0;k<3;++k) { c[k]=rh(rgb[3*i+k]); mn[k]=std::min(mn[k],c[k]); mx[k]=std::max(mx[k],c[k]); }
        l[q]=.25f*c[0]+.5f*c[1]+.25f*c[2]; lo=std::min(lo,l[q]); hi=std::max(hi,l[q]);
    }
    RobustPixel result{};
    int mi=weights.config.mv_dilate ? best:centre;
    result.motion={motion[2*mi],motion[2*mi+1]};
    result.depth=weights.config.depth_dilate ? near:rh(depth[centre]);
    if(weights.config.thin_lock) {
        float ridge=0;
        for(int a=0;a<4;++a) {
            float bright=std::min(l[4]-l[a],l[4]-l[8-a]);
            float dark=std::min(l[a]-l[4],l[8-a]-l[4]);
            ridge=std::max(ridge,std::max(bright,dark));
        }
        result.thin=rh(std::clamp((ridge/std::max(hi-lo,.001f)-.25f)/.75f,0.f,1.f));
    }
    float factor=weights.config.thin_lock && result.thin>.5f ? rh(1+weights.scalar("thin_slack")):1;
    for(int c=0;c<3;++c) result.slack[c]=rh(rh(rh(mx[c]-mn[c])*rh(std::abs(weights.scalar("box_slack"))))*factor);
    return result;
}
CoveragePixel coverage_pixel(const float* rgb,const float* previous,std::array<float,2> motion,
                             int h,int w,int y,int x,bool reset) {
    auto rh=[](float v) { return from_half(to_half(v)); };
    float u=(x+.5f)/w+motion[0],v=(y+.5f)/h+motion[1];
    reset=reset || u<0 || u>=1 || v<0 || v>=1;
    float gx=u*2-1,gy=v*2-1;
    int ix=int(std::nearbyint(std::clamp(((gx+1)*w-1)/2,0.f,float(w-1))));
    int iy=int(std::nearbyint(std::clamp(((gy+1)*h-1)/2,0.f,float(h-1))));
    const float* prev=previous+4*(iy*w+ix);
    float mn[3]={3e38f,3e38f,3e38f},mx[3]={-3e38f,-3e38f,-3e38f};
    for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        int at=std::clamp(y+dy,0,h-1)*w+std::clamp(x+dx,0,w-1);
        for(int c=0;c<3;++c) { float a=rh(rgb[3*at+c]); mn[c]=std::min(mn[c],a); mx[c]=std::max(mx[c],a); }
    }
    const float* cur=rgb+3*(y*w+x);
    float luma=.25f*rh(cur[0])+.5f*rh(cur[1])+.25f*rh(cur[2]);
    float span=.25f*rh(mx[0]-mn[0])+.5f*rh(mx[1]-mn[1])+.25f*rh(mx[2]-mn[2])+.02f;
    CoveragePixel out{};
    if(reset) out.state={luma,luma*luma,0,luma};
    else {
        out.evidence={rh(std::clamp(std::max(prev[1]-prev[0]*prev[0],0.f)/span,0.f,4.f)),
                      rh(std::clamp(prev[2]/span,0.f,4.f))};
        out.state={prev[0]+.25f*(luma-prev[0]),prev[1]+.25f*(luma*luma-prev[1]),
                   prev[2]+.25f*(std::abs(luma-prev[3])-prev[2]),luma};
    }
    return out;
}
AgePixel age_pixel(const float* previous,std::array<float,2> motion,
                   int h,int w,int y,int x,bool reset) {
    float u=(x+.5f)/w+motion[0],v=(y+.5f)/h+motion[1];
    reset=reset || u<0 || u>=1 || v<0 || v>=1;
    float gx=u*2-1,gy=v*2-1;
    int ix=int(std::nearbyint(std::clamp(((gx+1)*w-1)/2,0.f,float(w-1))));
    int iy=int(std::nearbyint(std::clamp(((gy+1)*h-1)/2,0.f,float(h-1))));
    float age=reset ? 0:previous[iy*w+ix];
    return {age,reset ? 0:std::min(age+1,32.f),from_half(to_half(std::log2(1+age)/5))};
}
float age_alpha(float alpha,float age,float alpha_min,bool reset) {
    if(reset) return 1;
    float limit=from_half(to_half(std::max(1/(1+age),std::clamp(alpha_min,0.f,1.f))));
    return std::min(alpha,limit);
}
float coverage_floor(float bias) {
    return from_half(to_half(1/(1+std::exp(-from_half(to_half(bias))))));
}
bool depth_reset(const Weights& w,bool offscreen,bool first,float prev,float mn,float mx,float osc_n) {
    if(offscreen || first) return true;
    if(!w.config.depth_test) return false;
    auto rh=[](float v) { return from_half(to_half(v)); };
    mn=rh(mn); mx=rh(mx);
    bool mismatch=prev<rh(mn*.9f) || prev>rh(mx*1.1f);
    if(!w.config.depth_soft) return mismatch;
    if(!w.config.depth_soft_osc) return false;
    // Reversed-Z: a larger value is nearer. Foreground departure overrides dither.
    bool dither=rh(std::clamp(osc_n,0.f,4.f))>std::clamp(w.scalar("soft_osc_threshold"),.05f,4.f);
    return prev>rh(mx*1.25f) || (mismatch && !dither);
}
float coverage_coefficient(float logit, float bias) {
    auto rh=[](float v) { return from_half(to_half(v)); };
    float floor=coverage_floor(bias);
    return rh(std::clamp(rh(rh(1/(1+std::exp(-rh(logit))))-floor)/rh(1-floor),0.f,1.f));
}

}
