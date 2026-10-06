#include "common.hlsli"
Texture2D<float4> previous : register(t0);
Texture2D<float4> color : register(t1);
Texture2D<float2> vectors : register(t2);
Texture2D<float> depth : register(t3);
Texture2D<float> previous_depth : register(t4);
Texture2D<float> exposure : register(t5);
Texture2D<float4> previous_coverage : register(t6);
RWBuffer<float> X : register(u0);
RWBuffer<float> hcl : register(u1);
RWBuffer<float> hraw : register(u2);
RWBuffer<float> cw : register(u3);
RWBuffer<float> reset_buffer : register(u4);
RWTexture2D<float> next_depth : register(u5);
RWTexture2D<float4> next_coverage : register(u6);
Texture2D<float2> previous_age : register(t7);
RWTexture2D<float2> next_age : register(u7);
float3 lr(int2 p) {
    float3 c = color.Load(int3(clamp(p, int2(0,0), int2(w-1,h-1)),0)).rgb;
    if (!model_space) { c = max(c,0) * exposure_scale(has_exposure ? exposure.Load(int3(0,0,0)) : 1); c = c / (1+c); }
    return float3(rh(c.r),rh(c.g),rh(c.b));
}
#ifdef FSRM_FAST
float3 lr_at(int2 p,float gain) {
    float3 c=max(color.Load(int3(p,0)).rgb,0)*gain;
    c=c/(1+c);
    return float3(rh(c.r),rh(c.g),rh(c.b));
}
float3 nearest_tap(float3 taps[9],int2 delta) {
    return delta.y<0 ? (delta.x<0 ? taps[0] : (delta.x==0 ? taps[1]:taps[2]))
         : delta.y==0 ? (delta.x<0 ? taps[3] : (delta.x==0 ? taps[4]:taps[5]))
                      : (delta.x<0 ? taps[6] : (delta.x==0 ? taps[7]:taps[8]));
}
float2 motion_tap(float2 taps[9],int2 p) {
    return p.y==0 ? (p.x==0 ? taps[0] : (p.x==1 ? taps[1]:taps[2]))
         : p.y==1 ? (p.x==0 ? taps[3] : (p.x==1 ? taps[4]:taps[5]))
                  : (p.x==0 ? taps[6] : (p.x==1 ? taps[7]:taps[8]));
}
#endif
float dep(int2 p) { p=clamp(p,int2(0,0),int2(w-1,h-1)); float d=depth.Load(int3(p,0)); return rh(inverted_depth ? d : 1-d); }
// Reversed-Z; strict greater preserves the first row-major tie.
int2 nearest_depth(int2 p) {
    p=clamp(p,int2(0,0),int2(w-1,h-1));
    int2 best=clamp(p-1,int2(0,0),int2(w-1,h-1)); float near=dep(best);
    [unroll] for(int dy=-1;dy<=1;dy++) [unroll] for(int dx=-1;dx<=1;dx++) {
        int2 at=clamp(p+int2(dx,dy),int2(0,0),int2(w-1,h-1)); float d=dep(at);
        if(d>near) { near=d; best=at; }
    }
    return best;
}
float thin_feature(int2 p) {
    float l[9],lo=3e38,hi=-3e38;
    [unroll] for(int dy=-1;dy<=1;dy++) [unroll] for(int dx=-1;dx<=1;dx++) {
        float3 c=lr(p+int2(dx,dy)); int q=(dy+1)*3+dx+1;
        l[q]=.25*c.r+.5*c.g+.25*c.b; lo=min(lo,l[q]); hi=max(hi,l[q]);
    }
    float ridge=0;
    [unroll] for(int a=0;a<4;a++) {
        float bright=min(l[4]-l[a],l[4]-l[8-a]),dark=min(l[a]-l[4],l[8-a]-l[4]);
        ridge=max(ridge,max(bright,dark));
    }
    return rh(saturate((ridge/max(hi-lo,.001)-.25)/.75));
}
float2 mv(int2 p) {
    p=clamp(p,int2(0,0),int2(w-1,h-1));
    if(mv_dilate) p=nearest_depth(p);
    if(display_motion) p=p*2+1;
    return vectors.Load(int3(p,0))*motion_scale-jitter_cancel;
}
// CUDA lines 463-494: independently clamped bilinear/bicubic taps, A=-.75.
float c1(float x) { return ((1.25*x-2.25)*x)*x+1; }
float c2(float x) { return ((-.75*x+3.75)*x-6)*x+3; }
float4 cubic(float t) { return float4(c2(t+1),c1(t),c1(1-t),c2(2-t)); }
float4 sample4(float2 p) {
    if(!bicubic) p=clamp(p,0,float2(2*w-1,2*h-1));
    int2 ip=int2(floor(p)); float2 t=p-ip;
    float4 wx=bicubic ? cubic(t.x) : float4(1-t.x,t.x,0,0);
    float4 wy=bicubic ? cubic(t.y) : float4(1-t.y,t.y,0,0);
    int count=bicubic ? 4:2, start=bicubic ? -1:0;
    int xs[4],ys[4];
    [unroll] for(int k=0;k<4;k++) { xs[k]=bound(ip.x+k+start,2*w); ys[k]=bound(ip.y+k+start,2*h); }
    float4 result=0;
#ifdef FSRM_FAST
    [unroll]
#else
    [loop]
#endif
    for(int j=0;j<count;j++) {
        float4 row=0;
#ifdef FSRM_FAST
        [unroll] for(int i=0;i<count;i++) row+=previous.Load(int3(xs[i],ys[j],0))*wx[i];
#else
        [loop] for(int i=0;i<count;i++) row+=previous.Load(int3(clamp(ip+int2(i,j)+start,int2(0,0),int2(2*w-1,2*h-1)),0))*wx[i];
#endif
        result+=row*wy[j];
    }
    return result;
}
// CUDA lines 495-510: bilinear motion in render-resolution UV units.
float2 motion(float2 p) {
    p=max(p,0); int2 ip=int2(floor(p)); float2 t=p-ip;
    float2 a=(1-t.x)*mv(ip)+t.x*mv(ip+int2(1,0));
    float2 b=(1-t.x)*mv(ip+int2(0,1))+t.x*mv(ip+int2(1,1));
    return (1-t.y)*a+t.y*b;
}
float log1p_(float x) { return x < 1e-3 ? x*(1 - x*(.5 - x*(1.0/3 - x*.25))) : log(1+x); }
void put(uint base,uint channel,uint phase,float v) { X[base+(channel*4+phase)*(hp/2)*(wp/2)]=rh(v); }
// CUDA lines 518-633: one thread per padded render pixel, NCHW space-to-depth.
[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID) {
    uint xp=id.x, yp=id.y; if(xp>=wp || yp>=hp) return;
    int x=bound(xp,w),y=bound(yp,h); uint at=y*w+x, n=h*w;
    uint base=(yp/2)*(wp/2)+xp/2, phase=(yp%2)*2+xp%2;
    bool valid=xp<w && yp<h;
#ifdef FSRM_FAST
    float gain=exposure_scale(has_exposure ? exposure.Load(int3(0,0,0)):1);
    int xs[3],ys[3];
    [unroll] for(int k=0;k<3;k++) { xs[k]=bound(x+k-1,w); ys[k]=bound(y+k-1,h); }
    float3 colors[9]; float depths[9]; float2 motions[9];
    [unroll] for(int dy=0;dy<3;dy++) [unroll] for(int dx=0;dx<3;dx++) {
        int2 p=int2(xs[dx],ys[dy]); int k=dy*3+dx;
        colors[k]=lr_at(p,gain);
        if(!mv_dilate) {
            float d=depth.Load(int3(p,0)); depths[k]=rh(inverted_depth ? d:1-d);
            int2 mp=display_motion ? p*2+1:p;
            motions[k]=vectors.Load(int3(mp,0))*motion_scale-jitter_cancel;
        }
    }
    if(mv_dilate) {
        float window[25];
        [unroll] for(int dy=0;dy<5;dy++) [unroll] for(int dx=0;dx<5;dx++)
            window[dy*5+dx]=dep(int2(x+dx-2,y+dy-2));
        [unroll] for(int dy=0;dy<3;dy++) [unroll] for(int dx=0;dx<3;dx++) {
            int2 local=int2(dx+1,dy+1), best=local-1;
            float near=window[dy*5+dx];
            [unroll] for(int sy=-1;sy<=1;sy++) [unroll] for(int sx=-1;sx<=1;sx++) {
                int2 at=local+int2(sx,sy); float d=window[at.y*5+at.x];
                if(d>near) { near=d; best=at; }
            }
            int2 mp=clamp(int2(x,y)+best-2,int2(0,0),int2(w-1,h-1));
            if(display_motion) mp=mp*2+1;
            motions[dy*3+dx]=vectors.Load(int3(mp,0))*motion_scale-jitter_cancel;
            depths[dy*3+dx]=window[(dy+1)*5+dx+1];
        }
        // mv() clamps its centre before searching. Border neighbours reuse that centre's result.
        [unroll] for(int k=0;k<3;k++) {
            if(x==0) motions[k*3]=motions[k*3+1];
            if(x==int(w)-1) motions[k*3+2]=motions[k*3+1];
        }
        [unroll] for(int k=0;k<3;k++) {
            if(y==0) motions[k]=motions[k+3];
            if(y==int(h)-1) motions[k+6]=motions[k+3];
        }
    }
    float2 m=motions[4];
    float2 uv=
#else
    float2 m=mv(int2(x,y)), uv=
#endif
    (float2(x,y)+.5)/float2(w,h)+m;
    float rs=(first_frame || any(uv<0) || any(uv>=1)) ? 1:0;
    float reset_feature=rs;
    bool mismatch=false, departure=false;
    if(depth_test && !first_frame) {
        float2 g=uv*2-1; g=float2(rh(g.x),rh(g.y));
        float2 pos=((g+1)*float2(w,h)-1)/2;
        int2 ip=int2(round(clamp(pos,0,float2(w-1,h-1))));
        float prev=previous_depth.Load(int3(ip,0)), mn=3e38,mx=-3e38;
        [unroll] for(int dy=-1;dy<=1;dy++) [unroll] for(int dx=-1;dx<=1;dx++) {
            int2 p=int2(x+dx,y+dy);
            if(all(p>=0) && p.x<int(w) && p.y<int(h)) {
#ifdef FSRM_FAST
                float v=depths[(dy+1)*3+dx+1];
#else
                float v=dep(p);
#endif
                mn=min(mn,v); mx=max(mx,v);
            }
        }
        mismatch=prev<rh(mn*.9) || prev>rh(mx*1.1);
        departure=prev>rh(mx*1.25);
        if(mismatch) {
            reset_feature=1;
            if(!depth_soft) rs=1;
        }
    }
    float3 mn=3e38,mx=-3e38;
#ifdef FSRM_FAST
    [unroll] for(int k=0;k<9;k++) { mn=min(mn,colors[k]); mx=max(mx,colors[k]); }
#else
    [unroll] for(int dy=-1;dy<=1;dy++) [unroll] for(int dx=-1;dx<=1;dx++) { float3 v=lr(int2(x+dx,y+dy)); mn=min(mn,v); mx=max(mx,v); }
#endif
    float thin=thin_lock ? thin_feature(int2(x,y)) : 0;
    if(thin_lock) put(base,25+(carry_raw ? 4:0),phase,thin);
#ifdef FSRM_FAST
    float3 lo,hi,ranges,current=colors[4];
#else
    float3 lo,hi,ranges,current=lr(int2(x,y));
#endif
    [unroll] for(int c=0;c<3;c++) {
        ranges[c]=rh(mx[c]-mn[c]); float slack=rh(ranges[c]*abs(box_slack));
        if(thin_lock) slack=rh(slack*(thin>.5 ? thin_factor : 1));
        lo[c]=rh(mn[c]-slack); hi[c]=rh(mx[c]+slack);
        put(base,c,phase,current[c]); put(base,15+c,phase,ranges[c]);
    }
#ifdef FSRM_FAST
    float d=depths[4];
#else
    float d=dep(int2(x,y));
#endif
    put(base,18,phase,rh(1/max(d,rh(.01))));
    float2 vel=m*float2(w,h); float speed=sqrt(vel.x*vel.x+vel.y*vel.y);
    put(base,19,phase,min(speed*.1,1)); put(base,20,phase,reset_feature);
    float conf_den=conf_motion ? rh(1+rh(abs(conf_m)*rh(min(speed,16)))) : 1;
    float luma_range=rh(rh(rh(.25*ranges[0])+rh(.5*ranges[1]))+rh(.25*ranges[2]));
    if(coverage) {
        // RGBA: m1, m2, oscillation, previous instantaneous luma, all float32.
        float2 g=uv*2-1, pos=((g+1)*float2(w,h)-1)/2;
        int2 ip=int2(round(clamp(pos,0,float2(w-1,h-1))));
        float4 prev=first_frame ? 0:previous_coverage.Load(int3(ip,0));
        float luma=.25*current.r+.5*current.g+.25*current.b;
        float span=.25*ranges.r+.5*ranges.g+.25*ranges.b+.02;
        if(depth_soft_osc) {
            // Pre-update, half-rounded evidence. Inference uses a strict hard step.
            // Reversed-Z: larger is nearer; foreground departure always resets.
            float osc_n=rs ? 0:rh(clamp(prev.z/span,0,4));
            bool dither=osc_n>clamp(soft_osc_threshold,.05,4);
            if(departure || (mismatch && !dither)) rs=1;
        }
        uint cov0=25+(carry_raw ? 4:0)+(thin_lock ? 1:0);
        put(base,cov0,phase,rs ? 0:clamp(max(prev.y-prev.x*prev.x,0)/span,0,4));
        put(base,cov0+1,phase,rs ? 0:clamp(prev.z/span,0,4));
        if(valid) next_coverage[int2(x,y)]=rs ? float4(luma,luma*luma,0,luma)
            : float4(prev.x+.25*(luma-prev.x),prev.y+.25*(luma*luma-prev.y),
                     prev.z+.25*(abs(luma-prev.w)-prev.z),luma);
    }
    if(history_age) {
        float2 g=uv*2-1, pos=((g+1)*float2(w,h)-1)/2;
        int2 ip=int2(round(clamp(pos,0,float2(w-1,h-1))));
        float age=rs ? 0:previous_age.Load(int3(ip,0)).x;
        uint age0=25+(carry_raw ? 4:0)+(thin_lock ? 1:0)+(coverage ? 2:0);
        put(base,age0,phase,log2(1+age)/5);
        // Keep the pre-update count beside the persisted count for resolve at the cap.
        if(valid) next_age[int2(x,y)]=float2(rs ? 0:min(age+1,32),age);
    }
    if(valid) { reset_buffer[at]=rs; next_depth[int2(x,y)]=depth_dilate ? dep(nearest_depth(int2(x,y))) : d; }
    [unroll] for(int q=0;q<4;q++) {
        int2 outp=int2(2*x+q%2,2*y+q/2);
#ifdef FSRM_FAST
        float2 pos_mv=max((outp+.5)/2-.5,0);
        int2 ip_mv=int2(floor(pos_mv)),local=ip_mv-int2(x,y)+1; float2 t_mv=pos_mv-ip_mv;
        float2 a=(1-t_mv.x)*motion_tap(motions,local)+t_mv.x*motion_tap(motions,local+int2(1,0));
        float2 b=(1-t_mv.x)*motion_tap(motions,local+int2(0,1))+t_mv.x*motion_tap(motions,local+int2(1,1));
        float2 mm=(1-t_mv.y)*a+t_mv.y*b;
#else
        float2 mm=motion((outp+.5)/2-.5);
#endif
        float2 g=((outp+.5)/float2(2*w,2*h)+mm)*2-1;
        float2 pos=((g+1)*float2(2*w,2*h)-1)/2;
        float4 sampled=first_frame ? 0 : sample4(pos);
        [unroll] for(int c=0;c<3;c++) {
            float hist=rh(sampled[c]),cl=clamp(hist,lo[c],hi[c]);
            put(base,3+c*4+q,phase,cl);
            if(valid) { hcl[(c*4+q)*n+at]=cl; if(carry_raw) hraw[(c*4+q)*n+at]=hist; }
        }
        float conf=rh(max(rh(sampled.w),0)*(1-rs)); if(conf_motion) conf=rh(conf/conf_den);
        put(base,21+q,phase,log1p_(conf)); if(valid) cw[q*n+at]=conf;
        if(carry_raw) {
            int2 delta=nearest_sample ? offsets[q].yx : int2(0,0);
#ifdef FSRM_FAST
            float3 b=nearest_tap(colors,delta);
#else
            float3 b=lr(int2(x,y)+delta);
#endif
            float raw_luma=rh(rh(rh(.25*rh(sampled.x))+rh(.5*rh(sampled.y)))+rh(.25*rh(sampled.z)));
            float base_luma=rh(rh(rh(.25*b.x)+rh(.5*b.y))+rh(.25*b.z));
            float innovation=rh(rh(raw_luma-base_luma)/rh(luma_range+.02));
            put(base,25+q,phase,rh(clamp(innovation,-4,4)*(1-rs)));
        }
    }
}
