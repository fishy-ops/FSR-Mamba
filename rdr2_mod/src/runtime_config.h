#pragma once
#include <algorithm>
#include <array>
#include <string>
#include <cmath>
#include <cwchar>
#include <cwctype>
#include <limits>

namespace fsrmamba {
inline float runtime_float(const wchar_t* text,float fallback,float minimum,float maximum) {
    wchar_t* end=nullptr;
    float value=std::wcstof(text,&end);
    if(end==text || !std::isfinite(value)) return fallback;
    while(std::iswspace(*end)) ++end;
    return *end ? fallback:std::clamp(value,minimum,maximum);
}
// INI zero selects the start only for parameters with a positive lower bound.
struct FoliageControls {
    float foliage_strength=0, foliage_ema=.2f, foliage_threshold=.08f, foliage_eps=.004f;
    float foliage_spatial=.3f, foliage_history_scale=.5f, foliage_alpha_floor=.1f;
    float fallback_strength=0, fallback_sigma=.65f, fallback_alpha=.5f;
    bool enabled() const { return foliage_strength>0 || fallback_strength>0; }
};
struct FoliageSetting { const wchar_t* name; const char* cli; float FoliageControls::*field; float start, minimum, maximum; };
inline constexpr FoliageSetting foliage_settings[]={
    {L"foliage_strength","--foliage",&FoliageControls::foliage_strength,0,0,1},
    {L"foliage_ema","--foliage-ema",&FoliageControls::foliage_ema,.2f,.02f,1},
    {L"foliage_threshold","--foliage-threshold",&FoliageControls::foliage_threshold,.08f,.01f,.5f},
    {L"foliage_eps","--foliage-eps",&FoliageControls::foliage_eps,.004f,.0001f,.1f},
    {L"foliage_spatial","--foliage-spatial",&FoliageControls::foliage_spatial,.3f,0,.6f},
    {L"foliage_history_scale","--foliage-history-scale",&FoliageControls::foliage_history_scale,.5f,.25f,1},
    {L"foliage_alpha_floor","--foliage-alpha-floor",&FoliageControls::foliage_alpha_floor,.1f,.05f,.5f},
    {L"fallback_strength","--fallback",&FoliageControls::fallback_strength,0,0,1},
    {L"fallback_sigma","--fallback-sigma",&FoliageControls::fallback_sigma,.65f,.3f,1},
    {L"fallback_alpha","--fallback-alpha",&FoliageControls::fallback_alpha,.5f,.2f,1},
};
inline float foliage_float(const wchar_t* text,const FoliageSetting& setting) {
    const float value=runtime_float(text,setting.start,-std::numeric_limits<float>::max(),setting.maximum);
    return value==0 && setting.minimum>0 ? setting.start:std::clamp(value,setting.minimum,setting.maximum);
}
inline int runtime_key(const wchar_t* text) noexcept {
    struct Key { const wchar_t* name; int code; };
    static constexpr Key keys[]={
        {L"VK_BACK",0x08},{L"VK_TAB",0x09},{L"VK_RETURN",0x0d},{L"VK_SHIFT",0x10},
        {L"VK_CONTROL",0x11},{L"VK_MENU",0x12},{L"VK_PAUSE",0x13},{L"VK_CAPITAL",0x14},
        {L"VK_ESCAPE",0x1b},{L"VK_SPACE",0x20},{L"VK_PRIOR",0x21},{L"VK_NEXT",0x22},
        {L"VK_END",0x23},{L"VK_HOME",0x24},{L"VK_LEFT",0x25},{L"VK_UP",0x26},
        {L"VK_RIGHT",0x27},{L"VK_DOWN",0x28},{L"VK_SNAPSHOT",0x2c},{L"VK_INSERT",0x2d},
        {L"VK_DELETE",0x2e},{L"VK_LWIN",0x5b},{L"VK_RWIN",0x5c},{L"VK_APPS",0x5d},
        {L"VK_MULTIPLY",0x6a},{L"VK_ADD",0x6b},{L"VK_SEPARATOR",0x6c},{L"VK_SUBTRACT",0x6d},
        {L"VK_DECIMAL",0x6e},{L"VK_DIVIDE",0x6f},{L"VK_NUMLOCK",0x90},{L"VK_SCROLL",0x91},
        {L"VK_LSHIFT",0xa0},{L"VK_RSHIFT",0xa1},{L"VK_LCONTROL",0xa2},{L"VK_RCONTROL",0xa3},
        {L"VK_LMENU",0xa4},{L"VK_RMENU",0xa5},{L"VK_OEM_1",0xba},{L"VK_OEM_PLUS",0xbb},
        {L"VK_OEM_COMMA",0xbc},{L"VK_OEM_MINUS",0xbd},{L"VK_OEM_PERIOD",0xbe},{L"VK_OEM_2",0xbf},
        {L"VK_OEM_3",0xc0},{L"VK_OEM_4",0xdb},{L"VK_OEM_5",0xdc},{L"VK_OEM_6",0xdd},
        {L"VK_OEM_7",0xde},{L"VK_OEM_8",0xdf},{L"VK_OEM_102",0xe2},
    };
    for(const auto& key:keys) if(std::wcscmp(text,key.name)==0) return key.code;
    if(std::wcsncmp(text,L"VK_NUMPAD",9)==0 && text[9]>=L'0' && text[9]<=L'9' && !text[10]) return 0x60+text[9]-L'0';
    wchar_t* end=nullptr;
    if(std::wcsncmp(text,L"VK_F",4)==0) {
        const long n=std::wcstol(text+4,&end,10);
        return end!=text+4 && !*end && n>=1 && n<=24 ? 0x70+int(n)-1:0;
    }
    const long n=std::wcstol(text,&end,0);
    while(std::iswspace(*end)) ++end;
    return end!=text && !*end && n>=1 && n<=255 ? int(n):0;
}
struct LiveControls {
    float stabilize=0,stabilize_tau=.5f,stabilize_eps=.004f;
    FoliageControls foliage;
};
struct StabilizeSetting { const wchar_t* name; float LiveControls::*field; float start, minimum, maximum; };
inline constexpr StabilizeSetting stabilize_settings[]={
    {L"stabilize",&LiveControls::stabilize,0,0,.95f},
    {L"stabilize_tau",&LiveControls::stabilize_tau,.5f,.05f,4},
    {L"stabilize_eps",&LiveControls::stabilize_eps,.004f,1e-4f,.1f},
};
inline constexpr size_t live_setting_count=3+std::size(foliage_settings);
inline const wchar_t* live_setting_name(size_t i) {
    return i<3 ? stabilize_settings[i].name:foliage_settings[i-3].name;
}
inline float& live_setting_value(LiveControls& controls,size_t i) {
    return i<3 ? controls.*(stabilize_settings[i].field):controls.foliage.*(foliage_settings[i-3].field);
}
inline void apply_live_setting(LiveControls& controls,size_t i,const wchar_t* text) {
    if(i<3) {
        const auto& s=stabilize_settings[i];
        live_setting_value(controls,i)=runtime_float(text,s.start,s.minimum,s.maximum);
    } else live_setting_value(controls,i)=foliage_float(text,foliage_settings[i-3]);
}
struct LivePreset {
    std::array<float,live_setting_count> values{};
    std::array<bool,live_setting_count> defined{};
    bool valid=false;
    void apply(LiveControls& controls) const noexcept {
        if(valid) for(size_t i=0;i<live_setting_count;++i)
            if(defined[i]) live_setting_value(controls,i)=values[i];
    }
};
enum class PresetParse { empty, valid, invalid };
inline PresetParse parse_preset(const std::wstring& text,LivePreset& preset) noexcept {
    preset={};
    try {
        auto trim=[](std::wstring value) {
            const auto first=value.find_first_not_of(L" \t\r\n");
            return first==std::wstring::npos ? std::wstring():value.substr(first,value.find_last_not_of(L" \t\r\n")-first+1);
        };
        if(trim(text).empty()) return PresetParse::empty;
        LivePreset parsed;
        LiveControls controls;
        size_t start=0;
        for(;;) {
            const auto comma=text.find(L',',start);
            const auto pair=text.substr(start,comma==std::wstring::npos ? comma:comma-start);
            const auto equal=pair.find(L'=');
            if(equal==std::wstring::npos) return PresetParse::invalid;
            const auto key=trim(pair.substr(0,equal)),value=trim(pair.substr(equal+1));
            size_t i=0;
            while(i<live_setting_count && key!=live_setting_name(i)) ++i;
            if(i==live_setting_count || value.empty()) return PresetParse::invalid;
            wchar_t* end=nullptr;
            const float number=std::wcstof(value.c_str(),&end);
            if(end==value.c_str() || *end || !std::isfinite(number)) return PresetParse::invalid;
            apply_live_setting(controls,i,value.c_str());
            parsed.values[i]=live_setting_value(controls,i); parsed.defined[i]=true;
            if(comma==std::wstring::npos) break;
            start=comma+1;
        }
        parsed.valid=true; preset=parsed;
        return PresetParse::valid;
    } catch(...) { return PresetParse::invalid; }
}
struct PresetCycle {
    std::array<LivePreset,6> presets{};
    int active=-1;
    void ini_reload(bool changed) noexcept { if(changed) active=-1; }
    int advance() noexcept {
        for(int step=1;step<=6;++step) {
            const int next=(active+step)%6;
            if(presets[next].valid) { active=next; return next+1; }
        }
        return 0;
    }
    LiveControls apply(const LiveControls& ini) const noexcept {
        auto controls=ini;
        if(active>=0) presets[active].apply(controls);
        return controls;
    }
};

}
