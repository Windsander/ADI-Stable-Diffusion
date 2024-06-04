/*
 * Copyright (c) 2018-2050 Word Piece Encoding(WP) Tokenizer - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef TOKENIZER_WP_H
#define TOKENIZER_WP_H

#include "tokenizer_base.cc"

namespace onnx {
namespace sd {
namespace tokenizer {

class WPTokenizer : public TokenizerBase {
protected:
    const std::string bep_vocab_end = ",";
    const std::regex bep_split_reg = std::regex(
        R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|\d|[^ \t\n\r\f\v\w]+)",
        std::regex::icase
    );

protected:
    std::tuple<Tokens, Multis, size_t> encode(PromptWeight_map prompt_weight_) override {

        const float token_end_multi_ = get_boundary_factor();
        const int token_end_index_ = get_end_token_index();
        const int token_safe_gaps_ = 20;
        const int avail_ = get_avail_token_size();      // limit of current token_size 75

        Tokens remade_tokens;
        Multis remade_multis;

        size_t pair_count_ = 1;
        int last_vocab_at_ = -1;
        for (auto concise_: prompt_weight_) {
            std::vector<std::string> vocab = PromptsHelper::split(
                PromptsHelper::whitespace(concise_.first),
                bep_split_reg, false
            );
            for (const std::string& character_: vocab) {
                bool reach_space_mark_ = (character_ == bep_vocab_end);
                bool needs_split_last_ = ((remade_tokens.size() % avail_ == 0) && (last_vocab_at_ != -1) &&
                                          (remade_tokens.size() - last_vocab_at_ <= token_safe_gaps_));
                if (reach_space_mark_) {
                    last_vocab_at_ = remade_tokens.size();
                } else if (needs_split_last_) {
                    last_vocab_at_ += 1;
                    Tokens tokens_cache_(remade_tokens.begin() + last_vocab_at_, remade_tokens.end());
                    Multis multis_cache_(remade_multis.begin() + last_vocab_at_, remade_multis.end());

                    // do split token with last reach max length
                    remade_tokens.resize(last_vocab_at_);
                    remade_multis.resize(last_vocab_at_);
                    int token_end_ = ceil(float(remade_tokens.size()) / float(avail_)) * avail_ - remade_tokens.size();
                    remade_tokens.insert(remade_tokens.end(), token_end_, token_end_index_);
                    remade_multis.insert(remade_multis.end(), token_end_, token_end_multi_);

                    remade_tokens.insert(remade_tokens.end(), tokens_cache_.begin(), tokens_cache_.end());
                    remade_multis.insert(remade_multis.end(), multis_cache_.begin(), multis_cache_.end());
                    pair_count_ += 1;
                }

                remade_tokens.push_back(sd_tokenizer_tok2id[character_ + "</w>"]);
                remade_multis.push_back(concise_.second);
            }
        }

        int finish_at_ = ceil(remade_tokens.size() / float(avail_)) * avail_ - remade_tokens.size();
        remade_tokens.insert(remade_tokens.end(), finish_at_, token_end_index_);
        remade_multis.insert(remade_multis.end(), finish_at_, token_end_multi_);

        return {remade_tokens, remade_multis, pair_count_};
    }

public:
    explicit WPTokenizer(const TokenizerConfig &tokenizer_config_ = {}) : TokenizerBase(tokenizer_config_) {};
    ~WPTokenizer() override = default;
};

} // namespace tokenizer
} // namespace sd
} // namespace onnx

#endif //TOKENIZER_WP_H

