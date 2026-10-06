#pragma once
#include <windows.h>
#include <bcrypt.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

namespace fsrmamba {
inline std::string file_hash(const std::filesystem::path& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()<0 || f.tellg()>(1LL<<30)) throw std::runtime_error("cannot hash file");
    std::vector<unsigned char> data(static_cast<size_t>(f.tellg())); f.seekg(0);
    if(!f.read(reinterpret_cast<char*>(data.data()),data.size())) throw std::runtime_error("hash read failed");
    BCRYPT_ALG_HANDLE algorithm=nullptr;
    if(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0)<0) throw std::runtime_error("SHA256 unavailable");
    unsigned char digest[32];
    auto status=BCryptHash(algorithm,nullptr,0,data.data(),static_cast<ULONG>(data.size()),digest,32);
    BCryptCloseAlgorithmProvider(algorithm,0);
    if(status<0) throw std::runtime_error("SHA256 failed");
    std::string out; for(auto b:digest) { out+="0123456789abcdef"[b>>4]; out+="0123456789abcdef"[b&15]; } return out;
}
}
