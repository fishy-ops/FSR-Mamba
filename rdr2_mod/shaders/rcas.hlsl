// This file is part of the FidelityFX SDK.
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Five-tap FsrRcasF limiters from FSR 2.2.1; denoising is disabled.
#include "common.hlsli"
Texture2D<float4> display : register(t0);
Texture2D<float> exposure : register(t1);
RWTexture2D<float4> output : register(u0);
cbuffer Pass : register(b1) { uint exposure_reset; float rcas_linear; };

float max_rgb(float3 c) { return max(c.r,max(c.g,c.b)); }
float3 prepare(int2 p,float scale) {
    float3 c=max(display.Load(int3(clamp(p,int2(0,0),int2(2*w-1,2*h-1)),0)).rgb,0);
    if(model_space) { c=min(c,1-1.0/1024); c=c/(1-c); }
    c=min(c*scale,65504);
    return c/(1+max_rgb(c));
}

[numthreads(8,8,1)]
void main(uint3 id:SV_DispatchThreadID) {
    if(id.x>=2*w || id.y>=2*h) return;
    int2 p=int2(id.xy);
    float scale=model_space ? 1 : exposure_scale(has_exposure ? exposure.Load(int3(0,0,0)) : 1);
    float3 b=prepare(p+int2(0,-1),scale),d=prepare(p+int2(-1,0),scale);
    float3 e=prepare(p,scale),f=prepare(p+int2(1,0),scale),h_=prepare(p+int2(0,1),scale);
    float3 mn=min(min(b,d),min(f,h_)),mx=max(max(b,d),max(f,h_));
    // An all-zero channel leaves the upper limiter in control, without a 0/0.
    float3 hit_min;
    [unroll] for(int channel=0;channel<3;channel++) hit_min[channel]=mx[channel]>0 ? mn[channel]/(4*mx[channel]) : 1;
    float3 hit_max=(1-mx)/min(4*mn-4,-1e-8);
    float3 limits=max(-hit_min,hit_max);
    // User sharpness S -> stops 2*(1-S) -> exp2(-stops), with the requested peak mapping.
    float peak=-1/lerp(8.0,5.0,rcas_linear);
    float lobe=max(peak,min(max_rgb(limits),0));
    float3 c=(lobe*b+lobe*d+lobe*h_+lobe*f+e)/(4*lobe+1);
    c=max(c,0);
    c=c/max(1-max_rgb(c),1.0/32768);
    if(model_space) c=c/(1+c);
    else c=c/scale;
    output[id.xy]=float4(c,1);
}
