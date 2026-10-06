#include <cstddef>
#include <cstdint>
#include "ffx_fsr2.h"
static_assert(sizeof(void*)==8 && sizeof(wchar_t)==2);
static_assert(sizeof(FfxResource)==0xb8);
static_assert(offsetof(FfxResource,description)==0x88);
static_assert(offsetof(FfxResource,state)==0xa4);
static_assert(offsetof(FfxResource,isDepth)==0xa8);
static_assert(offsetof(FfxResource,descriptorData)==0xb0);
static_assert(sizeof(FfxFsr2ContextDescription)==0x98);
static_assert(offsetof(FfxFsr2ContextDescription,callbacks)==0x18);
static_assert(offsetof(FfxFsr2ContextDescription,device)==0x88);
static_assert(offsetof(FfxFsr2ContextDescription,fpMessage)==0x90);
static_assert(sizeof(FfxFsr2DispatchDescription)==0x618);
static_assert(offsetof(FfxFsr2DispatchDescription,color)==8);
static_assert(offsetof(FfxFsr2DispatchDescription,renderSize)==0x520);
static_assert(offsetof(FfxFsr2DispatchDescription,reset)==0x538);
static_assert(offsetof(FfxFsr2DispatchDescription,enableAutoReactive)==0x54c);
static_assert(offsetof(FfxFsr2DispatchDescription,colorOpaqueOnly)==0x550);
static_assert(offsetof(FfxFsr2DispatchDescription,autoTcThreshold)==0x608);
int main() { return 0; }
