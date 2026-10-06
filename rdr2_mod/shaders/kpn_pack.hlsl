#include "common.hlsli"
Texture2D<float4> previous : register(t0);
Texture2D<float4> color : register(t1);
Texture2D<float2> vectors : register(t2);
Texture2D<float> depth : register(t3);
Texture2D<float> previous_depth : register(t4);
Texture2D<float> exposure : register(t5);
Texture2D<float4> previous_tracker : register(t6);
Texture2D<float2> previous_age : register(t7);
RWBuffer<float> X : register(u0);
RWTexture2D<float4> reprojected : register(u1);
RWTexture2D<float4> next_tracker : register(u2);
SamplerState history_sampler : register(s0);
RWTexture2D<float> next_depth : register(u5);
RWTexture2D<float2> next_age : register(u7);

int2 clamp_lr(int2 p) { return clamp(p,int2(0,0),int2(w-1,h-1)); }
float3 lr(int2 p) {
    float3 c=color.Load(int3(clamp_lr(p),0)).rgb;
    if(!model_space) { c=max(c,0)*exposure_scale(has_exposure ? exposure.Load(int3(0,0,0)):1); c=c/(1+c); }
    return float3(rh(c.r),rh(c.g),rh(c.b));
}
float3 ycocg(float3 c) { return float3(.25*c.r+.5*c.g+.25*c.b,.5*c.r-.5*c.b,-.25*c.r+.5*c.g-.25*c.b); }
float dep(int2 p) {
    p=clamp_lr(p); if(kpn_depth_hr) p*=2;
    float d=depth.Load(int3(p,0)); return inverted_depth ? d:1-d;
}
float2 raw_motion(int2 p) { return vectors.Load(int3(p,0))*motion_scale-jitter_cancel; }
groupshared float tile_luma[12*12];
groupshared float tile_depth[12*12];
groupshared float2 tile_motion[12*12];
groupshared float2 selected_motion[10*10];
static int2 group_origin;
uint source_index(int2 p) {
    int2 local=clamp(p-(group_origin-2),int2(0,0),int2(11,11));
    return local.y*12+local.x;
}
void cache_motion(uint index) {
    for(uint i=index;i<12*12;i+=64) {
        int2 p=clamp_lr(group_origin-2+int2(i%12,i/12));
        tile_depth[i]=dep(p);
        if(foliage_strength>0 || fallback_strength>0) tile_luma[i]=ycocg(lr(p)).x;
        float2 m;
        if(display_motion) {
            p*=2;
            m=(raw_motion(p)+raw_motion(p+int2(1,0))+raw_motion(p+int2(0,1))+raw_motion(p+1))*.25;
        } else m=raw_motion(p);
        tile_motion[i]=m;
    }
    GroupMemoryBarrierWithGroupSync();
    for(uint i=index;i<10*10;i+=64) {
        int2 p=clamp_lr(group_origin-1+int2(i%10,i/10));
        uint best=source_index(p);
        if(mv_dilate) {
            best=source_index(p-1); float near=tile_depth[best];
            [unroll] for(int dy=-1;dy<=1;dy++) [unroll] for(int dx=-1;dx<=1;dx++) {
                uint at=source_index(p+int2(dx,dy)); float d=tile_depth[at];
                if(d>near) { near=d; best=at; }
            }
        }
        selected_motion[i]=tile_motion[best];
    }
    GroupMemoryBarrierWithGroupSync();
}
float2 mv(int2 p) {
    int2 local=clamp(clamp_lr(p)-(group_origin-1),int2(0,0),int2(9,9));
    return selected_motion[local.y*10+local.x];
}
float2 motion_hr(int2 p) {
    if(display_motion && !mv_dilate) return raw_motion(p);
    float2 pos=(float2(p)+.5)*.5-.5;
    int2 ip=int2(floor(pos)); float2 t=pos-ip;
    float2 a=(1-t.x)*mv(ip)+t.x*mv(ip+int2(1,0));
    float2 b=(1-t.x)*mv(ip+int2(0,1))+t.x*mv(ip+1);
    return (1-t.y)*a+t.y*b;
}
float c1(float x) { return ((1.25*x-2.25)*x)*x+1; }
float c2(float x) { return ((-.75*x+3.75)*x-6)*x+3; }
float4 cubic(float t) { return float4(c2(t+1),c1(t),c1(1-t),c2(2-t)); }
float2 grid_position(float2 uv,float2 size) { return (((uv*2-1)+1)*size-1)*.5; }
float4 catmull(float t) {
    float t2=t*t,t3=t2*t;
    return float4(-.5*t+t2-.5*t3,1-2.5*t2+1.5*t3,.5*t+2*t2-1.5*t3,-.5*t2+.5*t3);
}
float3 history_catmull(float2 uv) {
    float2 size=float2(2*w,2*h),p=uv*size-.5,ip=floor(p),t=p-ip;
    float4 wx=catmull(t.x),wy=catmull(t.y);
    float3 ax=float3(wx.x,wx.y+wx.z,wx.w),ay=float3(wy.x,wy.y+wy.z,wy.w);
    float3 px=float3(ip.x-1,ip.x+wx.z/ax.y,ip.x+2);
    float3 py=float3(ip.y-1,ip.y+wy.z/ay.y,ip.y+2);
    float3 result=0;
    // Only the two positive middle lobes can share a bilinear fetch.
    [unroll] for(uint j=0;j<3;j++) {
        float3 row=0;
        [unroll] for(uint i=0;i<3;i++)
            row+=previous.SampleLevel(history_sampler,(float2(px[i],py[j])+.5)/size,0).rgb*ax[i];
        result+=row*ay[j];
    }
    return clamp(result,0,1-1e-6);
}
float3 history(float2 uv) {
    if(kpn_catmull) return history_catmull(uv);
    float2 p=grid_position(uv,float2(2*w,2*h));
    int2 ip=int2(floor(p)); float2 t=p-ip;
    float4 wx=cubic(t.x),wy=cubic(t.y); float3 result=0;
    [unroll] for(int j=0;j<4;j++) {
        float3 row=0;
        [unroll] for(int i=0;i<4;i++) row+=previous.Load(int3(clamp(ip+int2(i-1,j-1),int2(0,0),int2(2*w-1,2*h-1)),0)).rgb*wx[i];
        result+=row*wy[j];
    }
    return clamp(result,0,1-1e-6);
}
float input_luma(int2 p) {
    p=clamp_lr(p);
    int2 local=p-(group_origin-2);
    if(any(local<0) || any(local>=12)) return ycocg(lr(p)).x;
    return tile_luma[local.y*12+local.x];
}
float unjittered_luma(float2 xy) {
    int2 p=int2(floor(xy)); float2 t=xy-p;
    return lerp(lerp(input_luma(p),input_luma(p+int2(1,0)),t.x),
                lerp(input_luma(p+int2(0,1)),input_luma(p+1),t.x),t.y);
}
float track(int2 p,float2 uv,int2 prev,bool invalid) {
    float L=unjittered_luma(float2(p)+signed_jitter),B=0,mn=3e38,mx=-3e38;
    [unroll] for(int dy=-1;dy<=1;dy++) [unroll] for(int dx=-1;dx<=1;dx++) {
        float y=unjittered_luma(float2(p+int2(dx,dy))+signed_jitter);
        B+=y*(dx==0 ? .5:.25)*(dy==0 ? .5:.25); mn=min(mn,y); mx=max(mx,y);
    }
    float d=0,I=0,confidence=0;
    [branch] if(!tracker_reset && !invalid) {
        float old_depth=previous_depth.Load(int3(prev,0)),z=dep(p);
        // A nearest, depth-consistent delta preserves its sign at silhouettes.
        if(abs(old_depth-z)<=.1*max(z,.001)) {
            float4 old=previous_tracker.Load(int3(prev,0));
            float old_B=previous_tracker.SampleLevel(history_sampler,uv,0).w;
            float agreement=1-smoothstep(.03,.06,abs(B-old_B));
            float spread2=0; float2 m=mv(p);
            [unroll] for(int j=-1;j<=1;j++) [unroll] for(int i=-1;i<=1;i++) {
                float2 delta=(mv(p+int2(i,j))-m)*float2(w,h);
                spread2=max(spread2,dot(delta,delta));
            }
            confidence=agreement*saturate(1-sqrt(spread2)/1.5);
            d=(L-B)-(old.x-old.w);
            float event=d*old.y < -foliage_eps*foliage_eps ? saturate(min(abs(d),abs(old.y))/(mx-mn+foliage_eps)):0;
            I=lerp(old.z,event,foliage_ema)*agreement;
            if(agreement==0) d=0;
        }
    }
    next_tracker[p]=float4(L,d,I,B);
    return confidence;
}
void put(uint at,uint channel,float value) { X[channel*hp*wp+at]=value; }
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID,uint3 group:SV_GroupID,uint index:SV_GroupIndex) {
    group_origin=min(int2(group.xy*8),int2(w-1,h-1));
    cache_motion(index);
    if(id.x>=wp || id.y>=hp) return;
    int2 p=clamp_lr(int2(id.xy));
    uint stride=kpn_trunk_stride,phase=(id.y%stride)*stride+id.x%stride;
    uint base=phase*(hp/stride)*(wp/stride)+(id.y/stride)*(wp/stride)+id.x/stride;
    bool store=id.x<w && id.y<h;
    float2 m=mv(p),uv=(float2(p)+.5)/float2(w,h)+m;
    bool offscreen=any(uv<0) || any(uv>=1);
    int2 prev=int2(round(clamp(grid_position(uv,float2(w,h)),0,float2(w-1,h-1))));
    float dmin=3e38,dmax=-3e38,lmin=3e38,lmax=-3e38;
    [unroll] for(int dy=-1;dy<=1;dy++) [unroll] for(int dx=-1;dx<=1;dx++) {
        int2 tap=p+int2(dx,dy); float d=dep(tap),y=ycocg(lr(tap)).x;
        dmin=min(dmin,d); dmax=max(dmax,d); lmin=min(lmin,y); lmax=max(lmax,y);
    }
    float old_depth=first_frame ? 1:previous_depth.Load(int3(prev,0));
    bool mismatch=old_depth<.9*dmin || old_depth>1.1*dmax;
    bool invalid=first_frame || mismatch || offscreen;
    bool reset=first_frame || offscreen || (!depth_soft && mismatch);
    float3 mean=0;
    [unroll] for(uint q=0;q<4;q++) {
        int2 dst=2*p+int2(q%2,q/2);
        float2 uvh=(float2(dst)+.5)/float2(2*w,2*h)+motion_hr(dst);
        bool outside=any(uvh<0) || any(uvh>=1),rs=reset || outside;
        invalid=invalid || outside;
        float3 hist=0;
        if(!rs) hist=history(uvh);
        float3 yh=ycocg(hist); mean+=yh*.25;
        put(base,6+q,yh.x);
        if(store) {
            reprojected[dst]=float4(hist,rs ? 1:0);
        }
    }
    float age=invalid ? 0:clamp(previous_age.Load(int3(prev,0)).x,0,32);
    float3 current=ycocg(lr(p));
    [unroll] for(uint c=0;c<3;c++) { put(base,c,current[c]); put(base,3+c,mean[c]); }
    put(base,10,clamp((mean.x-current.x)/(lmax-lmin+.02),-4,4));
    put(base,11,invalid ? 1:0);
    put(base,12,saturate(length(m*float2(w,h))/10));
    put(base,13,log2(1+age)/5);
    put(base,14,signed_jitter.x); put(base,15,signed_jitter.y);
    if(store) {
        float confidence=0;
        if(foliage_strength>0 || fallback_strength>0) confidence=track(p,uv,prev,invalid);
        next_depth[p]=dep(p); next_age[p]=float2(invalid ? 0:min(age+1,32),confidence);
    }
}
