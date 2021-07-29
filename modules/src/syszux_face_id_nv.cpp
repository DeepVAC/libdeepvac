#include <cmath>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <sstream>
#include "NvInfer.h"
#include "gemfield.h"
#include "syszux_face_id_nv.h"

namespace deepvac {
SyszuxFaceIdNV::SyszuxFaceIdNV(std::string path, std::string device):DeepvacNV(path, device) {
    setBinding(2);
    initBinding();
}

SyszuxFaceIdNV::SyszuxFaceIdNV(std::vector<unsigned char>&& buffer, std::string device):DeepvacNV(std::move(buffer), device){
    setBinding(2);
    initBinding();
}

void SyszuxFaceIdNV::initBinding() {
    datas_[0].hostBuffer.resize({1, 3, 112, 112});
    datas_[0].deviceBuffer.resize({1, 3, 112, 112});
    datas_[1].hostBuffer.resize({1, 512});
    datas_[1].deviceBuffer.resize({1, 512});
}


void SyszuxFaceIdNV::loadDB(std::string path){
    try{
        std::ifstream inputstream(path + ".id");
        std::copy(std::istream_iterator<std::string>(inputstream),std::istream_iterator<std::string>(),std::back_inserter(id_vec_));
        torch::load(db_,path + ".db", device_);
    }catch(...){
        throw std::runtime_error("invalid parameter path sepcified.");
    }
    //db_ = db_.to(getDevice());
    std::stringstream feature_size;
    feature_size << db_.sizes();
    std::string msg = gemfield_org::format("%s: %s | %s : %d", "load db size: ", feature_size.str().c_str(), "load name vector size: ", id_vec_.size());
    GEMFIELD_I(msg.c_str());
}

int SyszuxFaceIdNV::size(){
    return id_vec_.size();
}

int SyszuxFaceIdNV::add(cv::Mat& frame, std::string name){
    cv::Mat dst;
    frame.convertTo(dst, CV_32F, 1.0 / 127.5, -1.0);                                                                                      
    float* input_data = (float*)dst.data;                                                                                               
    float* hostDataBuffer = static_cast<float*>(datas_[0].hostBuffer.data());                                                           
    for (int c = 0; c < 3; ++c) {                                                                                                       
        for(int j = 0, volChl=112*112; j < volChl; ++j) {                                                                               
            hostDataBuffer[c*volChl + j] = input_data[j*3 + c];                                                                         
        }                                                                                                                               
    }                                                                                                                                   
    cudaMemcpy(datas_[0].deviceBuffer.data(), datas_[0].hostBuffer.data(), datas_[0].hostBuffer.nbBytes(), cudaMemcpyHostToDevice);     
    std::vector<void*> predicitonBindings = {datas_[0].deviceBuffer.data(), datas_[1].deviceBuffer.data()};                             
    auto predict = forward(predicitonBindings.data());                                                                                  
    cudaMemcpy(datas_[1].hostBuffer.data(), datas_[1].deviceBuffer.data(), datas_[1].deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost);   
    auto emb = torch::from_blob(datas_[1].hostBuffer.data(), {1, 512}).to(device_);
    db2commit_vec_.push_back(emb);
    id2commit_vec_.push_back(name);
    return id2commit_vec_.size() - 1;
}

int SyszuxFaceIdNV::cachedSize(){
    return id2commit_vec_.size();
}

void SyszuxFaceIdNV::commit(std::string path_prefix){
    torch::Tensor db = torch::cat(db2commit_vec_, 0);
    std::stringstream feature_size;
    feature_size << db.sizes();
    std::string msg = gemfield_org::format("commit feature size: %s | commit name vector size: %d", "", feature_size.str().c_str(), id2commit_vec_.size() );
    GEMFIELD_I(msg.c_str());

    std::ofstream name_output(path_prefix + ".id");
    for (auto &name : id2commit_vec_){
        name_output << name << "\n";
    }
    torch::save(db , path_prefix + ".db");
    db2commit_vec_.clear();
    id2commit_vec_.clear();
}



std::optional<std::vector<std::tuple<int, std::string, float>>> SyszuxFaceIdNV::process(std::vector<cv::Mat>& frames) {
    if (frames.size()==0) {
        return std::nullopt;
    }
    
    std::vector<at::Tensor> embs_vec;
    cv::Mat dst;
    for(auto& frame : frames) {
        frame.convertTo(dst, CV_32F, 1.0 / 127.5, -1.0);
        float* input_data = (float*)dst.data;
        float* hostDataBuffer = static_cast<float*>(datas_[0].hostBuffer.data());
        for (int c = 0; c < 3; ++c) {
            for(int j = 0, volChl=112*112; j < volChl; ++j) {
                hostDataBuffer[c*volChl + j] = input_data[j*3 + c];
            }
        }
        cudaMemcpy(datas_[0].deviceBuffer.data(), datas_[0].hostBuffer.data(), datas_[0].hostBuffer.nbBytes(), cudaMemcpyHostToDevice);
        std::vector<void*> predicitonBindings = {datas_[0].deviceBuffer.data(), datas_[1].deviceBuffer.data()};
        auto predict = forward(predicitonBindings.data());
        cudaMemcpy(datas_[1].hostBuffer.data(), datas_[1].deviceBuffer.data(), datas_[1].deviceBuffer.nbBytes(), cudaMemcpyDeviceToHost);
        auto emb = torch::from_blob(datas_[1].hostBuffer.data(), {1, 512}).to(device_);
        embs_vec.emplace_back(std::move(emb));
    }

    std::vector<std::tuple<int, std::string, float>> results;
    for (auto embs : embs_vec) {
        for (int i=0; i<embs.size(0); i++) {
            auto emb = embs[i];
            at::Tensor distance = torch::norm(db_ - emb, 2,  1);
            int min_index = torch::argmin(distance).item<int64_t>();
            float min_distance = torch::min(distance).item<float>();
            GEMFIELD_DI2("predict: ", id_vec_.at(min_index).c_str());
            auto tmp_tuple = std::make_tuple(min_index, id_vec_.at(min_index), min_distance);
            results.push_back(tmp_tuple);
        }
    }
    return results;

}

}//namespace
