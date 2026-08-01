/*
 * Copyright (c) 2018-2050 SD_Tokenizer
 * Created by Arikan.Li on 2026/08/01.
 *
 * SentencePiece Tokenizer (T5-XXL, SD3.5/FLUX text_encoder_3)
 * base on: google/sentencepiece official C++ library (vendored, static)
 *
 * T5 layout semantics (diffusers T5TokenizerFast parity):
 *   sequence = [t1 ... tk, </s>=1, <pad>=0 ...]   total = avail_token_size (256)
 *   - NO leading start mark (unlike CLIP's <|startoftext|>)
 *   - </s> appended once per chunk; pads are id 0
 *   - prompts longer than (avail_token_size - 1) are chunked like the CLIP
 *     path (each chunk carries its own </s>), embedding pairs are merged
 *     along the sequence dim by the Clip unit
 *   - empty prompt -> [</s>, pads...] (diffusers negative-prompt behavior)
 *
 * Boundary note: this tokenizer overrides tokenize() wholesale instead of the
 * base's start/end wrapping, because T5's layout has no start mark and the
 * base assembly is CLIP-specific.
 */
#ifndef TOKENIZER_ENCODE_SP_H
#define TOKENIZER_ENCODE_SP_H

#include "tokenizer_base.cc"
#include "sentencepiece_processor.h"

namespace onnx {
namespace sd {
namespace tokenizer {

class SPTokenizer : public TokenizerBase {
private:
    sentencepiece::SentencePieceProcessor spm_;

protected:
    // segments -> ids, chunked at (avail_token_size - 1) with <pad>=0 fills
    std::tuple<Tokens, Multis, size_t> encode(PromptWeight_map prompt_weight_) override {
        const int chunk_size_ = get_avail_token_size() + 1;    // 255 for T5-256
        const int pad_index_ = 0;                              // T5 <pad>
        const float pad_multi_ = get_boundary_factor();

        Tokens remade_tokens;
        Multis remade_multis;

        for (auto concise_ : prompt_weight_) {
            std::vector<int32_t> ids_ = spm_.EncodeAsIds(concise_.first);
            for (int32_t id_ : ids_) {
                remade_tokens.push_back(id_);
                remade_multis.push_back(concise_.second);
            }
        }
        if (remade_tokens.empty()) {
            // give tokenize() one chunk to work with
            remade_tokens.push_back(pad_index_);
            remade_multis.push_back(pad_multi_);
        }

        // pad the tail of the last chunk so every chunk is chunk_size_ long
        size_t pair_count_ = (remade_tokens.size() + size_t(chunk_size_) - 1) / size_t(chunk_size_);
        size_t finish_at_ = pair_count_ * size_t(chunk_size_) - remade_tokens.size();
        remade_tokens.insert(remade_tokens.end(), finish_at_, pad_index_);
        remade_multis.insert(remade_multis.end(), finish_at_, pad_multi_);

        return {remade_tokens, remade_multis, pair_count_};
    }

public:
    explicit SPTokenizer(const TokenizerConfig &tokenizer_config_ = {}) : TokenizerBase(tokenizer_config_) {};
    ~SPTokenizer() override = default;

    void init() override;
    void uninit() override;

    // T5 layout: [t1..tk, </s>, <pad>...]; base CLIP-style start/end wrap is skipped
    PreparedToken_vec tokenize(const std::string &prompts_) override;
};

void SPTokenizer::init() {
    // sd_tokenizer_config.tokenizer_dictionary_at points at spiece.model
    auto status_ = spm_.Load(sd_tokenizer_config.tokenizer_dictionary_at);
    if (!status_.ok()) {
        amon_report(class_exception(EXC_LOG_ERR,
            ("ERROR:: sentencepiece model load failed: " + status_.ToString()).c_str()));
    }
}

void SPTokenizer::uninit() {
}

TokenizerBase::PreparedToken_vec SPTokenizer::tokenize(const std::string &prompts_) {
    constexpr int32_t SP_EOS_INDEX = 1;    // T5 </s>
    constexpr int32_t SP_PAD_INDEX = 0;    // T5 <pad>
    const int chunk_size_ = get_avail_token_size() + 1;    // 255 for T5-256

    PreparedToken_vec matched_results_;
    auto emit_pair_ = [&](const Tokens &chunk_tokens_, const Multis &chunk_multis_) {
        Tokens tokens_cache_(chunk_tokens_);
        Multis multis_cache_(chunk_multis_);
        tokens_cache_.push_back(SP_EOS_INDEX);
        multis_cache_.push_back(get_boundary_factor());
        size_t pad_at_ = size_t(sd_tokenizer_config.avail_token_size) - tokens_cache_.size();
        tokens_cache_.insert(tokens_cache_.end(), pad_at_, SP_PAD_INDEX);
        multis_cache_.insert(multis_cache_.end(), pad_at_, get_boundary_factor());

        TensorShape paired_shape_ = {1, sd_tokenizer_config.avail_token_size};
        Tensor token_tensor = TensorHelper::create<int32_t>(paired_shape_, tokens_cache_);
        Tensor multi_tensor = TensorHelper::create<float>(paired_shape_, multis_cache_);
        matched_results_.emplace_back(std::move(token_tensor), std::move(multi_tensor));
    };

    if (prompts_.empty()) {
        // empty prompt -> [</s>, pads...] (diffusers T5 negative-prompt parity)
        emit_pair_({}, {});
        return matched_results_;
    }

    PromptWeight_map cur_parsed_attention = parse_prompt_attention(prompts_);
    std::tuple<Tokens, Multis, size_t> encoded_input = encode(cur_parsed_attention);

    Tokens &encoded_tokens_ = std::get<0>(encoded_input);
    Multis &encoded_multis_ = std::get<1>(encoded_input);
    size_t encoded_pair_num_ = std::get<2>(encoded_input);

    for (size_t p_ = 0; p_ < encoded_pair_num_; ++p_) {
        Tokens chunk_tokens_(encoded_tokens_.begin() + p_ * chunk_size_,
                             encoded_tokens_.begin() + (p_ + 1) * chunk_size_);
        Multis chunk_multis_(encoded_multis_.begin() + p_ * chunk_size_,
                             encoded_multis_.begin() + (p_ + 1) * chunk_size_);
        // strip the encode()-time pad fills at chunk tail so </s> sits right
        // after the real tokens (pads follow it afterwards)
        while (!chunk_tokens_.empty() && chunk_tokens_.back() == SP_PAD_INDEX &&
               chunk_tokens_.size() > 1) {
            chunk_tokens_.pop_back();
            chunk_multis_.pop_back();
        }
        emit_pair_(chunk_tokens_, chunk_multis_);
    }

    return matched_results_;
}

} // namespace tokenizer
} // namespace sd
} // namespace onnx

#endif //TOKENIZER_ENCODE_SP_H
