/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */

#pragma once

#include <istream>
#include <streambuf>

class SyszuxStreamBuffer : public std::streambuf
{
    public:
        SyszuxStreamBuffer(const uint8_t *begin, const size_t size);
        SyszuxStreamBuffer() = delete;
        SyszuxStreamBuffer(const SyszuxStreamBuffer&) = delete;
        SyszuxStreamBuffer& operator=(const SyszuxStreamBuffer&) = delete;
        SyszuxStreamBuffer(SyszuxStreamBuffer&&) = default;
        SyszuxStreamBuffer& operator=(SyszuxStreamBuffer&&) = default;

    private:
        int_type underflow();
        int_type uflow();
        int_type pbackfail(int_type ch);
        std::streamsize showmanyc();
        std::streampos seekoff ( std::streamoff off, std::ios_base::seekdir way,std::ios_base::openmode which = std::ios_base::in | std::ios_base::out );
        std::streampos seekpos ( std::streampos sp,std::ios_base::openmode which = std::ios_base::in | std::ios_base::out);

    private:
        const uint8_t * const begin_;
        const uint8_t * const end_;
        const uint8_t * current_;
};