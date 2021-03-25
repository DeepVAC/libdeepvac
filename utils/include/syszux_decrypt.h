/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <openssl/evp.h>
#include <openssl/aes.h>
#include <openssl/err.h>
#include <openssl/sha.h>
#include <openssl/md5.h>
#include "gemfield.h"

namespace gemfield_org{

    inline std::vector<unsigned char> readFileToVectorByte(const std::string& file_path){
        std::ifstream syszux_file(file_path, std::ios::binary | std::ios::ate);
        std::vector<unsigned char> vector_bytes;

        if (!syszux_file.eof() && !syszux_file.fail()) {
            syszux_file.seekg(0, std::ios_base::end);
            std::streampos file_size = syszux_file.tellg();
            vector_bytes.resize(file_size);
            syszux_file.seekg(0, std::ios_base::beg);
            syszux_file.read((char*)&vector_bytes[0], file_size);
        } else {
            std::string msg = gemfield_org::format("Can't read file at:  %s",file_path);
            GEMFIELD_E(msg.c_str());
            throw std::runtime_error(msg);
        }
        return vector_bytes;
    }

    inline std::string md5(std::string& data){
        unsigned char md[MD5_DIGEST_LENGTH];
        unsigned long data_len = data.size();
        MD5(reinterpret_cast<const unsigned char*>(data.c_str()), data_len, md);
        
        char result[MD5_DIGEST_LENGTH * 2 + 1];
        memset(result, 0, sizeof result);
        for (int i = 0; i < MD5_DIGEST_LENGTH; i++){
            sprintf(&result[i*2], "%02x", md[i]);
        }
        
        return std::string(result);
    }

    inline std::string md5(const char* data){
        std::string tmp(data);
        return md5(tmp);
    }

    using DeepvacKeyBytes = std::array<unsigned char, SHA256_DIGEST_LENGTH>;
    class SyszuxDecrypt{
        public:
            SyszuxDecrypt(){
                ctx_ = EVP_CIPHER_CTX_new();
                GEMFIELD_I("construct SyszuxDecrypt succeeded.");
            };
            SyszuxDecrypt(const SyszuxDecrypt&) = delete;
            SyszuxDecrypt& operator=(const SyszuxDecrypt&) = delete;
            SyszuxDecrypt(SyszuxDecrypt&&) = default;
            SyszuxDecrypt& operator=(SyszuxDecrypt&&) = default;
            SyszuxDecrypt(std::string key):key_(key){
                ctx_ = EVP_CIPHER_CTX_new();
                GEMFIELD_I("construct SyszuxDecrypt succeeded.");
            }
            virtual ~SyszuxDecrypt(){
                EVP_CIPHER_CTX_cleanup(ctx_);
                EVP_CIPHER_CTX_free(ctx_);
                GEMFIELD_I("destruct SyszuxDecrypt succeeded."); 
            }

            std::vector<unsigned char> de(const std::string file_path, const std::string k) {
                GEMFIELD_SI;
                key_ = k;
                DeepvacKeyBytes key = calculateSha256(key_);
                auto pt_bytes = readFileToVectorByte(file_path);

                GEMFIELD_I2("SYSZUX_DE version: ", OPENSSL_VERSION_TEXT);
                if (pt_bytes.size() <= AES_BLOCK_SIZE) {
                    std::string msg = gemfield_org::format("Input encrypted content size = %d is too small for AES CBC decryption.", file_path);
                    GEMFIELD_E(msg.c_str());
                    throw std::runtime_error(msg);
                }
                
                const std::vector<unsigned char> iv(pt_bytes.begin(), pt_bytes.begin() + AES_BLOCK_SIZE);

                EVP_CIPHER_CTX_init(ctx_);
                status_ = EVP_DecryptInit_ex(ctx_, EVP_aes_256_cbc(), nullptr, key.data(), iv.data());
                isThrowThisMsg("[SYSZUX_DE] INIT Error");

                int p_len = (int)pt_bytes.size();
                int f_len = 0;
                std::vector<unsigned char> result_bytes(p_len);
                status_ = EVP_DecryptUpdate(ctx_, result_bytes.data(), &p_len, pt_bytes.data() + AES_BLOCK_SIZE, (int)pt_bytes.size() - AES_BLOCK_SIZE);
                isThrowThisMsg("[SYSZUX_DE] UPDATE Error");
                
                status_ = EVP_DecryptFinal_ex(ctx_, result_bytes.data() + p_len, &f_len);
                isThrowThisMsg("[SYSZUX_DE] FINAL Error");
                
                result_bytes.resize(p_len + f_len);
                return result_bytes;
            }

        private:
            DeepvacKeyBytes calculateSha256(std::string key) {
                DeepvacKeyBytes key_bytes;
                SHA256_CTX sha256;
                SHA256_Init(&sha256);
                SHA256_Update(&sha256, key.c_str(), key.size());
                SHA256_Final(key_bytes.data(), &sha256);
                return key_bytes;
            }
            void isThrowThisMsg(std::string msg){
                if(status_ != 0){
                    return;
                }
                ERR_print_errors_fp(stderr);
                GEMFIELD_E(msg.c_str());
                throw std::runtime_error(msg);
            }

        private:
            std::string key_;
            EVP_CIPHER_CTX* ctx_;
            int status_{1};
    };
}
