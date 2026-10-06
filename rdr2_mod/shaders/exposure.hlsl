#include "common.hlsli"
Texture2D<float4> color : register(t0);
RWTexture2D<float> exposure : register(u0);
cbuffer Pass : register(b1) { uint exposure_reset; float rcas_linear; };
groupshared float log_luminance[64];

[numthreads(64,1,1)]
void main(uint index:SV_GroupIndex) {
    float sum=0;
    [loop] for(uint sample=index;sample<32*18;sample+=64) {
        uint2 cell=uint2(sample%32,sample/32);
        int2 p=min(int2((float2(cell)+.5)*float2(w,h)/float2(32,18)),int2(w-1,h-1));
        float3 c=max(color.Load(int3(p,0)).rgb,0);
        sum+=log(max(dot(c,float3(.2126,.7152,.0722)),1e-4));
    }
    log_luminance[index]=sum;
    GroupMemoryBarrierWithGroupSync();
    [unroll] for(uint stride=32;stride>0;stride/=2) {
        if(index<stride) log_luminance[index]+=log_luminance[index+stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if(index==0) {
        float target=1/(9.6*exp(log_luminance[0]/(32*18)));
        float value=target;
        if(!exposure_reset) value=lerp(exposure[uint2(0,0)],target,.1);
        exposure[uint2(0,0)]=clamp(value,1e-3,1e3);
    }
}
