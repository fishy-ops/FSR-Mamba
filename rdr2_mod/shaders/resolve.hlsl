#include "common.hlsli"
Buffer<float> o : register(t0);
Texture2D<float4> color : register(t1);
Buffer<float> hcl : register(t2);
Buffer<float> hraw : register(t3);
Buffer<float> cw : register(t4);
Buffer<float> reset_buffer : register(t5);
Texture2D<float> exposure : register(t6);
Texture2D<float2> next_age : register(t7);
RWTexture2D<float4> history : register(u0);
RWTexture2D<float4> output : register(u1);
float3 lr(int2 p) {
    float3 c=color.Load(int3(clamp(p,int2(0,0),int2(w-1,h-1)),0)).rgb;
    if(!model_space) { c=max(c,0)*exposure_scale(has_exposure ? exposure.Load(int3(0,0,0)) : 1); c=c/(1+c); }
    return float3(rh(c.r),rh(c.g),rh(c.b));
}
#ifdef FSRM_FAST
float3 lr_at(int2 p,float gain) {
    float3 c=max(color.Load(int3(p,0)).rgb,0)*gain;
    c=c/(1+c);
    return float3(rh(c.r),rh(c.g),rh(c.b));
}
float3 nearest_tap(float3 taps[16],int2 delta) {
    return delta.y<0 ? (delta.x<0 ? taps[5] : (delta.x==0 ? taps[6]:taps[7]))
         : delta.y==0 ? (delta.x<0 ? taps[9] : (delta.x==0 ? taps[10]:taps[11]))
                      : (delta.x<0 ? taps[13] : (delta.x==0 ? taps[14]:taps[15]));
}
// Constant window origins keep the 4x4 taps in registers. Tap order matches the 16-tap sum.
#define LANCZOS(NAME,WY,WX) \
float3 NAME(float3 taps[16],uint q) { \
    float3 acc=0,lo=taps[WY*4+WX],hi=lo; \
    [unroll] for(int dy=0;dy<3;dy++) [unroll] for(int dx=0;dx<3;dx++) { \
        float3 tap=taps[(WY+dy)*4+WX+dx]; \
        acc+=tap*rh(kernels[q*4+WY+dy][WX+dx]); lo=min(lo,tap); hi=max(hi,tap); \
    } \
    return clamp(float3(rh(acc.r),rh(acc.g),rh(acc.b)),lo,hi); \
}
LANCZOS(lanczos00,0,0)
LANCZOS(lanczos01,0,1)
LANCZOS(lanczos10,1,0)
LANCZOS(lanczos11,1,1)
float3 lanczos_phase(float3 taps[16],uint q) {
    switch(windows[q].x*2+windows[q].y) {
        case 0: return lanczos00(taps,q);
        case 1: return lanczos01(taps,q);
        case 2: return lanczos10(taps,q);
        default: return lanczos11(taps,q);
    }
}
#endif
// CUDA lines 636-720: phase blend, confidence, deringed base, carried vs displayed RGB.
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID) {
    uint x=id.x,y=id.y; if(x>=w || y>=h) return;
    uint at=y*w+x,n=h*w,stride=(hp/2)*(wp/2);
    uint base=(y/2)*(wp/2)+x/2+((y%2)*2+x%2)*stride;
    uint gate0=16,beta0=gate0+base_gate*4,cov0=beta0+carry_raw*4,keep0=cov0+(coverage ? 4:0);
    float3 taps[16];
#ifdef FSRM_FAST
    float gain=exposure_scale(has_exposure ? exposure.Load(int3(0,0,0)):1);
    int xs[4],ys[4];
    [unroll] for(int i=0;i<4;i++) { xs[i]=bound(int(x)+i-2,w); ys[i]=bound(int(y)+i-2,h); }
    [unroll] for(int ty=0;ty<4;ty++) [unroll] for(int tx=0;tx<4;tx++) taps[ty*4+tx]=lr_at(int2(xs[tx],ys[ty]),gain);
    bool is_reset=reset_buffer[at]>.5;
#else
    if(base_gate) [unroll] for(int t=0;t<16;t++) taps[t]=lr(int2(x,y)+int2(t%4-2,t/4-2));
#endif
    [unroll] for(int q=0;q<4;q++) {
        float logit=o[base+(12+q)*4*stride],keep=sig(o[base+(keep0+q)*4*stride]);
        float wh=rh(cw[q*n+at]*keep),weight=phase_weight[q];
        float cov=0;
        if(coverage) {
            float floor=coverage_floor;
            cov=rh(saturate(rh(sig(o[base+(cov0+q)*4*stride])-floor)/rh(1-floor)));
            weight=rh(weight*rh(1-rh(.9*cov)));
        }
        if(conf_consistent) { weight=rh(weight*rh(exp(clamp(logit,-2,2)))); logit=rh(rh(log(weight))-rh(log(rh(wh+.001)))); }
        else logit=rh(rh(logit+rh(log(weight)))-rh(log(rh(wh+.001))));
        float conf=min(rh(wh+weight),conf_max);
#ifdef FSRM_FAST
        float alpha=is_reset ? 1:sig(logit);
#else
        float alpha=reset_buffer[at]>.5 ? 1:sig(logit);
#endif
        if(history_age && reset_buffer[at]==0)
            alpha=min(alpha,rh(max(1/(1+next_age.Load(int3(x,y,0)).y),alpha_min)));
        float beta=carry_raw ? sig(o[base+(beta0+q)*4*stride]):0;
        float gate=base_gate ? sig(o[base+(gate0+q)*4*stride]):0;
        if(coverage) gate=rh(gate*rh(1-cov));
        int2 delta=nearest_sample ? offsets[q].yx:int2(0,0);
#ifdef FSRM_FAST
        float3 current=nearest_tap(taps,delta), carried,displayed,histv;
        float3 lanczos_base=lanczos_phase(taps,q);
#else
        float3 current=lr(int2(x,y)+delta), carried,displayed,histv;
#endif
        [unroll] for(int c=0;c<3;c++) {
            float residual=o[base+(c*4+q)*4*stride];
            if(base_gate) {
#ifdef FSRM_FAST
                current[c]=stable_lerp(lanczos_base[c],current[c],gate);
#else
                float lanczos=0;
                [unroll] for(int t=0;t<16;t++) lanczos+=taps[t][c]*rh(kernels[q*4+t/4][t%4]);
                int wy=windows[q].x,wx=windows[q].y; float lo=taps[wy*4+wx][c],hi=lo;
                [unroll] for(int dy=0;dy<3;dy++) [unroll] for(int dx=0;dx<3;dx++) { float tap=taps[(wy+dy)*4+wx+dx][c]; lo=min(lo,tap); hi=max(hi,tap); }
                lanczos=clamp(rh(lanczos),lo,hi); current[c]=stable_lerp(lanczos,current[c],gate);
#endif
            }
            float cur=carry_raw ? current[c]:rh(current[c]+residual);
            float hist=hcl[(c*4+q)*n+at];
            if(carry_raw) hist=stable_lerp(hraw[(c*4+q)*n+at],hist,beta);
            histv[c]=hist;
            float value=stable_lerp(hist,cur,alpha);
            carried[c]=value; displayed[c]=max(carry_raw ? rh(value+residual):value,0);
        }
        uint2 dst=uint2(x*2+q%2,y*2+q/2);
        history[dst]=float4(carried,conf);
        // debug 1: base only (Lanczos/nearest, no history); debug 2: reprojected history only.
        if(debug_view==1) displayed=current;
        if(debug_view==2) displayed=max(histv,0);
#ifdef FSRM_FAST
        displayed=clamp(displayed,0,1-1.0/1024); displayed=displayed/(1-displayed)/gain;
#else
        if(!model_space) { displayed=clamp(displayed,0,1-1.0/1024); displayed=displayed/(1-displayed)/exposure_scale(has_exposure ? exposure.Load(int3(0,0,0)):1); }
#endif
        output[dst]=float4(displayed,1);
    }
}
