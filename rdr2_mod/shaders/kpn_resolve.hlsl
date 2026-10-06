#include "common.hlsli"
#ifndef FSRM_KPN_TAPS
#define FSRM_KPN_TAPS 5
#endif
#define kpn_taps FSRM_KPN_TAPS
#define kpn_sigma_min phase_weight.x
#define kpn_proximity phase_weight.y
#define kpn_proximity_gain phase_weight.z
Buffer<float> params : register(t0);
Texture2D<float4> color : register(t1);
Texture2D<float4> reprojected : register(t2);
Buffer<float> packed : register(t3);
Texture2D<float4> tracker : register(t4);
Texture2D<float> tracker_depth : register(t5);
Texture2D<float> exposure : register(t6);
Texture2D<float2> age_confidence : register(t7);
RWTexture2D<float4> history : register(u0);
RWTexture2D<float4> output : register(u1);
float3 lr(int2 p) {
    float3 c=color.Load(int3(clamp(p,int2(0,0),int2(w-1,h-1)),0)).rgb;
    if(!model_space) { c=max(c,0)*exposure_scale(has_exposure ? exposure.Load(int3(0,0,0)):1); c=c/(1+c); }
    return float3(rh(c.r),rh(c.g),rh(c.b));
}
float sigmoid(float x) {
    float e=exp(-abs(x));
    return x>=0 ? 1/(1+e):e/(1+e);
}
groupshared float3 colors[14*14];
groupshared float2 tracker_tile[10*10];
static int2 tile_origin;
float3 cached_lr(int2 p) {
    int2 local=clamp(p,int2(0,0),int2(w-1,h-1))-tile_origin;
    // Captured jitter normally lies in [-.5,.5]; retain arbitrary-jitter behavior.
    if(any(local<0) || any(local>=14)) return lr(p);
    return colors[local.y*14+local.x];
}
float parameter(uint2 dst,uint c) {
    uint block=2*kpn_trunk_stride,tw=wp/kpn_trunk_stride,th=hp/kpn_trunk_stride;
    uint phase=(dst.y%block)*block+dst.x%block;
    uint channel=kpn_d2s_alt ? phase*7+c:c*block*block+phase;
    return params[channel*tw*th+(dst.y/block)*tw+dst.x/block];
}
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID,uint3 group:SV_GroupID,uint index:SV_GroupIndex) {
    tile_origin=int2(group.xy*8)-3;
    for(uint i=index;i<14*14;i+=64) colors[i]=lr(tile_origin+int2(i%14,i/14));
    if(foliage_strength>0 || fallback_strength>0) for(uint i=index;i<10*10;i+=64) {
        int2 p=clamp(int2(group.xy*8)-1+int2(i%10,i/10),int2(0,0),int2(w-1,h-1));
        tracker_tile[i]=float2(tracker.Load(int3(p,0)).z,tracker_depth.Load(int3(p,0)));
    }
    GroupMemoryBarrierWithGroupSync();
    if(id.x>=w || id.y>=h) return;
    float k=0,bad=0,confidence=0;
    [branch] if(foliage_strength>0 || fallback_strength>0) {
        uint stride=kpn_trunk_stride,tw=wp/stride,th=hp/stride;
        uint phase=(id.y%stride)*stride+id.x%stride;
        uint at=(11*stride*stride+phase)*tw*th+(id.y/stride)*tw+id.x/stride;
        bool valid=packed[at]<=.5 && !tracker_reset;
        int2 local=int2(id.xy-group.xy*8)+1;
        float z=tracker_tile[local.y*10+local.x].y,I=0,total=0,mn=3e38,mx=-3e38;
        [unroll] for(int dy=-1;dy<=1;dy++) [unroll] for(int dx=-1;dx<=1;dx++) {
            int2 p=clamp(int2(id.xy)+int2(dx,dy),int2(0,0),int2(w-1,h-1));
            float y=dot(cached_lr(p),float3(.25,.5,.25)); mn=min(mn,y); mx=max(mx,y);
            float weight=(dx==0 ? .5:.25)*(dy==0 ? .5:.25);
            float2 state=tracker_tile[(local.y+dy)*10+local.x+dx];
            if(abs(state.y-z)>.1*max(z,.001)) weight=0;
            I+=weight*state.x; total+=weight;
        }
        float HF=(mx-mn)/(mx-mn+foliage_eps);
        confidence=valid ? age_confidence.Load(int3(id.xy,0)).y:0;
        k=valid ? foliage_strength*HF*smoothstep(foliage_threshold,foliage_threshold+.1,I/max(total,1e-8)):0;
        bad=fallback_strength*HF*(1-confidence);
    }
    const int radius=kpn_taps/2,count=kpn_taps*kpn_taps;
    [loop] for(uint q=0;q<4;q++) {
        uint2 dst=id.xy*2+uint2(q%2,q/2);
        float2 sigma=max(exp(clamp(float2(parameter(dst,0),parameter(dst,1)),log(kpn_sigma_min),log(2.5))),1e-4);
        if(bad>0) sigma=lerp(sigma,max(sigma,fallback_sigma),bad);
        float angle=parameter(dst,2),cs=cos(angle),sn=sin(angle);
        float2 xy=(float2(dst)+.5)*.5-.5+signed_jitter;
        int2 centre=int2(clamp(floor(xy+.5),0,float2(w-1,h-1)));
        float logits[25],peak=-3e38;
        [unroll] for(int oy=-radius;oy<=radius;oy++) [unroll] for(int ox=-radius;ox<=radius;ox++) {
            int i=(oy+radius)*kpn_taps+ox+radius; int2 tap=centre+int2(ox,oy);
            float2 delta=float2(tap)-xy;
            float u=(cs*delta.x+sn*delta.y)/sigma.x,v=(-sn*delta.x+cs*delta.y)/sigma.y;
            bool valid=all(tap>=0) && tap.x<int(w) && tap.y<int(h);
            logits[i]=valid ? -.5*(u*u+v*v):-3e38; peak=max(peak,logits[i]);
        }
        float total=0,weights[25];
        [unroll] for(int i=0;i<count;i++) { weights[i]=exp(logits[i]-peak); total+=weights[i]; }
        float3 current=0;
        [unroll] for(int oy=-radius;oy<=radius;oy++) [unroll] for(int ox=-radius;ox<=radius;ox++)
            current+=(weights[(oy+radius)*kpn_taps+ox+radius]/max(total,1e-8))*cached_lr(centre+int2(ox,oy));
        float3 lo=3e38,hi=-3e38;
        [unroll] for(int dy=-1;dy<=1;dy++) [unroll] for(int dx=-1;dx<=1;dx++) {
            float3 v=cached_lr(centre+int2(dx,dy)); lo=min(lo,v); hi=max(hi,v);
        }
        float4 reproject=reprojected.Load(int3(dst,0));
        float3 hist=reproject.rgb;
        float slack=sigmoid(parameter(dst,4));
        float3 rectified=clamp(hist,lo-slack*(hi-lo),hi+slack*(hi-lo));
        float alpha=sigmoid(parameter(dst,3));
        if(kpn_proximity>0) {
            float2 delta=(float2(centre)-xy)*2;
            float r2=dot(delta,delta);
            if(bad>0) alpha=sigmoid(parameter(dst,3)+(log(kpn_proximity_gain)-r2/(2*kpn_proximity*kpn_proximity))*(1-bad));
            else alpha=sigmoid(parameter(dst,3)+log(kpn_proximity_gain)-r2/(2*kpn_proximity*kpn_proximity));
        }
        if(bad>0) alpha=lerp(alpha,max(alpha,fallback_alpha),bad);
        [branch] if(k>0) {
            [branch] if(foliage_spatial>0 && bad<1) {
                float3 spatial=0; float norm=0;
                float3 dxs=float3(-1,0,1)+(centre.x-xy.x),dys=float3(-1,0,1)+(centre.y-xy.y);
                float3 wx=exp(-dxs*dxs/(2*.65*.65)),wy=exp(-dys*dys/(2*.65*.65));
                [unroll] for(int dy=-1;dy<=1;dy++) [unroll] for(int dx=-1;dx<=1;dx++) {
                    int2 tap=centre+int2(dx,dy);
                    float weight=all(tap>=0) && tap.x<int(w) && tap.y<int(h) ? wx[dx+1]*wy[dy+1]:0;
                    spatial+=weight*cached_lr(tap); norm+=weight;
                }
                current=lerp(current,spatial/max(norm,1e-8),k*foliage_spatial*(1-bad));
            }
            alpha=lerp(alpha,min(alpha,max(foliage_alpha_floor,alpha*foliage_history_scale)),k*confidence*(1-bad));
        }
        if(reproject.a>.5) alpha=1;
        float y=parameter(dst,5),co=parameter(dst,6);
        float3 residual=kpn_residual ? .05*tanh(float3(y+co,y,y-co)):0;
        float3 value=clamp(alpha*current+(1-alpha)*rectified+residual,0,1-1e-6);
        if(bad>0 && reproject.a>.5) value=current;
        [branch] if(k==0 && bad==0 && stabilize>0 && !first_frame && reproject.a<=.5) {
            uint stride=kpn_trunk_stride,tw=wp/stride,th=hp/stride;
            uint phase=(id.y%stride)*stride+id.x%stride;
            // Feature 11 includes soft-depth disocclusion and all invalid output phases.
            uint at=(11*stride*stride+phase)*tw*th+(id.y/stride)*tw+id.x/stride;
            if(packed[at]<=.5) {
                float3 d=value-rectified;
                float L=dot(abs(d),float3(.2126,.7152,.0722));
                float R=dot(hi-lo,float3(.2126,.7152,.0722));
                float t=saturate(L/(stabilize_tau*R+stabilize_eps));
                float weight=stabilize*(1-t);
                if(weight>0) value=rectified+d*(1-weight);
            }
        }
        history[dst]=float4(value,0);
        float3 displayed=debug_view==1 ? current:debug_view==2 ? hist:value;
        if(!model_space) {
            displayed=clamp(displayed,0,1-1.0/1024);
            displayed=displayed/(1-displayed)/exposure_scale(has_exposure ? exposure.Load(int3(0,0,0)):1);
        }
        output[dst]=float4(displayed,1);
    }
}
