/*
 * Copyright (c) 2018-2050 SD_Tokenizer - Arikan.Li
 * Created by Arikan.Li on 2024/05/09.
 */
#ifndef TOKENIZER_REGISTER_ONCE
#define TOKENIZER_REGISTER_ONCE

#include "onnxsd_foundation.cc"
#include "tokenizer_base.cc"
#include "tokenizer_encode_bpe.cc"
#include "tokenizer_encode_wp.cc"

namespace onnx {
namespace sd {
namespace tokenizer {

using namespace base;

typedef TokenizerBase TokenizerEntity;
typedef TokenizerBase* TokenizerEntity_ptr;
typedef TokenizerBase::PreparedToken_vec PairedTokenWeight;

class TokenizerRegister {
public:
    static TokenizerEntity_ptr request_tokenizer(const TokenizerConfig &tokenizer_config_) {
        TokenizerEntity_ptr result_ptr_ = nullptr;
        switch (tokenizer_config_.tokenizer_type) {
            case TOKENIZER_BPE: {
                result_ptr_ = new BPETokenizer(tokenizer_config_);
                break;
            }
            case TOKENIZER_WORD_PIECE: {
                result_ptr_ = new WPTokenizer(tokenizer_config_);
                break;
            }
            default:{
                amon_report(class_exception(EXC_LOG_ERR, "ERROR:: selected Tokenizer unimplemented"));
                break;
            }
        }
        if (result_ptr_){
            result_ptr_->create();
        }
        return result_ptr_;
    }

    static TokenizerEntity_ptr recycle_tokenizer(TokenizerEntity_ptr target_ptr_){
        if (target_ptr_){
            target_ptr_->release();
            delete target_ptr_;
        }
        return nullptr;
    }
};

} // namespace tokenizer
} // namespace sd
} // namespace onnx

#endif  // TOKENIZER_REGISTER_ONCE