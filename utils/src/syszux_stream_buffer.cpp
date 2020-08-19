/*
 * Copyright (c) 2020 Gemfield <gemfield@civilnet.cn>
 * This file is part of libdeepvac, licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#include <cassert>
#include "syszux_stream_buffer.h"

SyszuxStreamBuffer::SyszuxStreamBuffer(const uint8_t *begin, const size_t size) :begin_(begin),end_(begin + size),current_(begin_)
{
    assert(std::less_equal<const uint8_t *>()(begin_, end_));
}

SyszuxStreamBuffer::int_type SyszuxStreamBuffer::underflow()
{
    if (current_ == end_)
        return traits_type::eof();

    return traits_type::to_int_type(*current_);
}

SyszuxStreamBuffer::int_type SyszuxStreamBuffer::uflow()
{
    if (current_ == end_)
        return traits_type::eof();

    return traits_type::to_int_type(*current_++);
}

SyszuxStreamBuffer::int_type SyszuxStreamBuffer::pbackfail(int_type ch)
{
    if (current_ == begin_ || (ch != traits_type::eof() && ch != current_[-1]))
        return traits_type::eof();

    return traits_type::to_int_type(*--current_);
}

std::streamsize SyszuxStreamBuffer::showmanyc()
{
    assert(std::less_equal<const uint8_t *>()(current_, end_));
    return end_ - current_;
}

std::streampos SyszuxStreamBuffer::seekoff ( std::streamoff off, std::ios_base::seekdir way, std::ios_base::openmode which )
{
    if (way == std::ios_base::beg){
        current_ = begin_ + off;
    }else if (way == std::ios_base::cur){
        current_ += off;
    }else if (way == std::ios_base::end){
        current_ = end_ + off;
    }

    if (current_ < begin_ || current_ > end_){
        return -1;
    }

    return current_ - begin_;
}

std::streampos SyszuxStreamBuffer::seekpos ( std::streampos sp, std::ios_base::openmode which )
{
    current_ = begin_ + sp;

    if (current_ < begin_ || current_ > end_){
        return -1;
    }

    return current_ - begin_;
}



