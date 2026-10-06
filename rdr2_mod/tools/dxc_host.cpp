// Minimal DXC library host for platforms without a packaged dxc executable.
#include <dxc/dxcapi.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

int main(int argc,char** argv) {
    if(argc!=4) { std::cerr<<"usage: dxc_host input.hlsl include-directory output.dxil\n"; return 1; }
    std::ifstream f(argv[1],std::ios::binary);
    if(!f) return 1;
    std::string source((std::istreambuf_iterator<char>(f)),{});
    std::string inc=argv[2]; std::wstring include(inc.begin(),inc.end());
    IDxcUtils* utils=nullptr; IDxcCompiler3* compiler=nullptr; IDxcIncludeHandler* handler=nullptr;
    if(FAILED(DxcCreateInstance(CLSID_DxcUtils,IID_PPV_ARGS(&utils))) || FAILED(DxcCreateInstance(CLSID_DxcCompiler,IID_PPV_ARGS(&compiler)))) return 1;
    if(FAILED(utils->CreateDefaultIncludeHandler(&handler))) return 1;
    const wchar_t* args[]={L"-T",L"cs_6_0",L"-E",L"main",L"-I",include.c_str(),L"-Gis",L"-WX",L"-O3"};
    DxcBuffer buffer{source.data(),source.size(),DXC_CP_UTF8}; IDxcResult* result=nullptr;
    if(FAILED(compiler->Compile(&buffer,args,sizeof(args)/sizeof(args[0]),handler,IID_PPV_ARGS(&result)))) return 1;
    IDxcBlobUtf8* errors=nullptr;
    result->GetOutput(DXC_OUT_ERRORS,IID_PPV_ARGS(&errors),nullptr);
    if(errors && errors->GetStringLength()) std::cerr<<errors->GetStringPointer();
    HRESULT status; result->GetStatus(&status); int rc=FAILED(status)?1:0;
    if(!rc) {
        IDxcBlob* obj=nullptr;
        if(FAILED(result->GetOutput(DXC_OUT_OBJECT,IID_PPV_ARGS(&obj),nullptr))) return 1;
        std::ofstream out(argv[3],std::ios::binary); out.write(static_cast<const char*>(obj->GetBufferPointer()),obj->GetBufferSize());
        if(!out) rc=1; obj->Release();
    }
    if(errors) errors->Release(); result->Release(); handler->Release(); compiler->Release(); utils->Release(); return rc;
}
