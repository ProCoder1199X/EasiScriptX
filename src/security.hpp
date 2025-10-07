#ifndef ESX_SECURITY_HPP
#define ESX_SECURITY_HPP

#include "config.hpp"
#include <string>
#include <vector>
#include <fstream>

#if USE_OPENSSL
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#endif

namespace esx::security {

inline bool read_file_bytes(const std::string& path, std::vector<unsigned char>& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<size_t>(size));
    return size >= 0 && f.read(reinterpret_cast<char*>(out.data()), size).good();
}

inline bool verify_signature(const std::string& payload_path,
                             const std::string& signature_path,
                             const std::string& public_key_pem_path) {
#if USE_OPENSSL
    std::vector<unsigned char> payload, signature;
    if (!read_file_bytes(payload_path, payload)) return false;
    if (!read_file_bytes(signature_path, signature)) return false;

    FILE* pub = fopen(public_key_pem_path.c_str(), "r");
    if (!pub) return false;
    EVP_PKEY* pkey = PEM_read_PUBKEY(pub, nullptr, nullptr, nullptr);
    fclose(pub);
    if (!pkey) return false;

    bool ok = false;
    EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
    if (mdctx && EVP_DigestVerifyInit(mdctx, nullptr, EVP_sha256(), nullptr, pkey) == 1) {
        if (EVP_DigestVerifyUpdate(mdctx, payload.data(), payload.size()) == 1) {
            ok = EVP_DigestVerifyFinal(mdctx, signature.data(), signature.size()) == 1;
        }
    }
    if (mdctx) EVP_MD_CTX_free(mdctx);
    EVP_PKEY_free(pkey);
    return ok;
#else
    (void)payload_path; (void)signature_path; (void)public_key_pem_path;
    return true; // No-op when OpenSSL is unavailable
#endif
}

} // namespace esx::security

#endif // ESX_SECURITY_HPP


