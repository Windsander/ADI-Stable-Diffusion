/*
 * Copyright (c) 2018-2050 Byte Pair Encoding(BEP) Tokenizer - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef TOKENIZER_BPE_H
#define TOKENIZER_BPE_H

#include "tokenizer_base.cc"

namespace onnx {
namespace sd {
namespace tokenizer {

class BPETokenizer : public TokenizerBase {
protected:
    std::string bep_pair_merge(const std::string &sentence_) {
        // before split sentence to words, we need do BPE first, that means do Merge First.
        // but first we need to check if merges.txt exist
        if (!sd_tokenizer_merge_ready) return sentence_;

        // step.1: cut sentence to chars, careful about space
        std::vector<std::string> char_list_(sentence_.size());
        std::transform(sentence_.begin(), sentence_.end(), char_list_.begin(), [](char c) {
            return std::string(1, c);
        });

        // step.2: search from bep_ranking_maps, found ranked byte(char)_pair and do merge for new one, update
        SubwordsPair_vec current_pairs_ = make_words_pair(char_list_);
        Subwords_pair minimum_pair = find_min_rank(current_pairs_);

        while (!(current_pairs_.empty()) &&
               !(minimum_pair.first.empty() && minimum_pair.second.empty())) {
            {
                std::vector<std::string> new_word;
                for (size_t i = 0; i < char_list_.size(); ++i) {
                    bool need_merge_ = (
                        i < char_list_.size() - 1 &&
                        char_list_[i] == minimum_pair.first &&
                        char_list_[i + 1] == minimum_pair.second
                    );
                    if (need_merge_) {  // merge & skip next
                        new_word.push_back(minimum_pair.first + minimum_pair.second);
                        ++i;
                    } else {            // keep steady
                        new_word.push_back(char_list_[i]);
                    }
                }
                char_list_ = std::move(new_word);
            }
            current_pairs_ = make_words_pair(char_list_);
            minimum_pair = find_min_rank(current_pairs_);
        }

        // step.3: if no chars can be merged, the return final result.
        // Mark: if there have unrecognisable words, that means words is not in vocabs, needs retraining add.
        std::string paired_result_;
        for (size_t i = 0; i < char_list_.size(); ++i) {
            paired_result_ += char_list_[i];
            if (i != char_list_.size() - 1) {
                paired_result_ += " ";
            }
        }
        return paired_result_;
    }

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

            std::string bep_paired_ = bep_pair_merge(concise_.first);

            std::vector<std::string> vocab_list_ = PromptsHelper::split(
                PromptsHelper::whitespace(bep_paired_),
                def_split_reg, false
            );
            for (const std::string& vocab_: vocab_list_) {
                bool reach_space_mark_ = (vocab_ == def_vocab_end);
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

                remade_tokens.push_back(sd_tokenizer_tok2id[vocab_ + "</w>"]);
                remade_multis.push_back(concise_.second);
            }
        }

        int finish_at_ = ceil(remade_tokens.size() / float(avail_)) * avail_ - remade_tokens.size();
        remade_tokens.insert(remade_tokens.end(), finish_at_, token_end_index_);
        remade_multis.insert(remade_multis.end(), finish_at_, token_end_multi_);

        return {remade_tokens, remade_multis, pair_count_};
    }

public:
    explicit BPETokenizer(const TokenizerConfig &tokenizer_config_ = {}) : TokenizerBase(tokenizer_config_) {};
    ~BPETokenizer() override = default;

    void init() override;
    void uninit() override;
};

void BPETokenizer::init(){
    // loading vocabulary
    load_vocab_file(sd_tokenizer_config.tokenizer_dictionary_at);
    // loading aggregates
    load_merge_file(sd_tokenizer_config.tokenizer_aggregates_at);
}

void BPETokenizer::uninit() {

}

} // namespace tokenizer
} // namespace sd
} // namespace onnx

#endif //TOKENIZER_BPE_H

