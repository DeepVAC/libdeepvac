/*
 * Copyright (c) 2020 gemfield <gemfield@civilnet.cn>
 * This file is part of libgemfield.so (https://github.com/civilnet/gemfield).
 * Licensed under the GPLv3 (the "License")
 * You may not use this file except in compliance with the License.
 */
#ifndef _GEMFIELD_H_
#define _GEMFIELD_H_

#include <initializer_list>
#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>
#include <map>
#include <fstream>

namespace gemfield_org{
    enum LOG_LEVEL{
        STACK_INFO = 0,
        DETAIL_INFO = 1,
        INFO = 2,
        WARNING = 5,
        ERROR = 6
    };
}
namespace gemfield_org{
    const  LOG_LEVEL global_log_level = DETAIL_INFO;
    class LogFromFile{
        public:
            LogFromFile(){
                std::cout<<"GEMFIELD INITIALIZATION ONLY ONCE!"<<std::endl;
                try{
                    std::ifstream infile("gemfield.loglevel");
                    std::string line;
                    while (std::getline(infile, line)) {

                        std::istringstream iss(line);
                        std::string k;
                        int v;
                        if (!(iss >> k >> v)) { 
                            break; 
                        } 
                        if(k == "LOGLEVEL"){
                            log_level_ = static_cast<LOG_LEVEL>(v);
                            break;
                        }
                    }
                } catch(...){
                    std::cout<<"Warning: read log configuration failed."<<std::endl;
                    log_level_ = global_log_level;
                }
            }
            LOG_LEVEL log_level_{global_log_level};
    };

    inline LOG_LEVEL getLogLevel(){
        static LogFromFile log_from_file;
        return log_from_file.log_level_;
    }

}

#define GEMFIELDSTR_DETAIL(x) #x
#define GEMFIELDSTR(x) GEMFIELDSTR_DETAIL(x)

#ifdef GARRULOUS_GEMFIELD
#define GEMFIELD_SI gemfield_org::Gemfield gemfieldsi({__FILE__, GEMFIELDSTR(__LINE__), __FUNCTION__}, gemfield_org::STACK_INFO)
#define GEMFIELD_DI(x) gemfield_org::Gemfield gemfielddi({__FILE__, GEMFIELDSTR(__LINE__), __FUNCTION__,x}, gemfield_org::DETAIL_INFO)
#define GEMFIELD_DI2(x,y) gemfield_org::Gemfield gemfielddi2({__FILE__, GEMFIELDSTR(__LINE__), __FUNCTION__,x,y}, gemfield_org::DETAIL_INFO)
#define GEMFIELD_I(x) gemfield_org::Gemfield gemfieldi({__FILE__, GEMFIELDSTR(__LINE__), __FUNCTION__,x}, gemfield_org::INFO)
#define GEMFIELD_I2(x,y) gemfield_org::Gemfield gemfieldi2({__FILE__, GEMFIELDSTR(__LINE__), __FUNCTION__,x,y}, gemfield_org::INFO)
#define GEMFIELD_W(x) gemfield_org::Gemfield gemfieldw({__FILE__, GEMFIELDSTR(__LINE__), __FUNCTION__,x}, gemfield_org::WARNING)
#define GEMFIELD_W2(x,y) gemfield_org::Gemfield gemfieldw2({__FILE__, GEMFIELDSTR(__LINE__), __FUNCTION__,x,y}, gemfield_org::WARNING)
#define GEMFIELD_E(x) gemfield_org::Gemfield gemfielde({__FILE__, GEMFIELDSTR(__LINE__), __FUNCTION__,x}, gemfield_org::ERROR)
#define GEMFIELD_E2(x,y) gemfield_org::Gemfield gemfielde2({__FILE__, GEMFIELDSTR(__LINE__), __FUNCTION__,x,y}, gemfield_org::ERROR)
#else
#define GEMFIELD_SI
#define GEMFIELD_DI(x)
#define GEMFIELD_DI2(x,y)
#define GEMFIELD_I(x)
#define GEMFIELD_I2(x,y)
#define GEMFIELD_W(x)
#define GEMFIELD_W2(x,y)
#define GEMFIELD_E(x)
#define GEMFIELD_E2(x,y)
#endif

thread_local int __attribute__((weak)) gemfield_counter = 0;
//std::map<uint64_t, int> __attribute__((weak)) gemfield_counter_map;
namespace gemfield_org{
    template<typename ... Args>
    std::string format( const std::string& format, Args ... args ){
        size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; 
        std::unique_ptr<char[]> buf( new char[ size ] ); 
        snprintf( buf.get(), size, format.c_str(), args ... );
        return std::string( buf.get(), buf.get() + size - 1 ); 
    }

    class Gemfield{
        public:
            Gemfield(std::initializer_list<const char*> src, LOG_LEVEL level):level_(level){
                if(level_ < getLogLevel()){
                    return;
                }
                std::stringstream ss;
                ss << "["<<std::this_thread::get_id()<<"]";
                for (auto s1: src){
                    ss << ":"<<s1;
                }
                s_ += ss.str();
                if(level_ != STACK_INFO){
                    printMark('#', s_, level_);
                }else{
                    printMark('+', s_, level_);
                }
            }
            ~Gemfield(){
                if(level_ < getLogLevel() || level_ != STACK_INFO){
                    return;
                }
                printMark('-', s_, level_);
            }

        private:
            static void printMark(char c, std::string& s, LOG_LEVEL level){
                static std::mutex gemfield_lock;
                std::lock_guard<std::mutex> lock(gemfield_lock);

                std::stringstream ss;
                ss << std::this_thread::get_id();
                uint64_t current_tid = std::stoull(ss.str());

                if(c == '+'){
                    ++gemfield_counter;
                }
                for(int i=0; i< gemfield_counter; i++){
                    std::cout<<c;
                }
                static std::map<LOG_LEVEL, std::string> log_token = {{STACK_INFO,""},{DETAIL_INFO," | DETAIL_INFO | "},{INFO," | INFO | "},{WARNING," | WARNING | "},{ERROR," | ERROR | "}};
                std::cout<<s<<log_token[level]<<std::endl;

                if(c == '-'){
                    --gemfield_counter;
                }
            }
            std::string s_;
            LOG_LEVEL level_{WARNING};
    };
}
#endif
