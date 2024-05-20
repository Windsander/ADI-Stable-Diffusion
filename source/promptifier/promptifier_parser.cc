/*
 * Copyright (c) 2018-2050 SD_Clip - Arikan.Li
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef MODEL_CLIP_H
#define MODEL_CLIP_H

#include "model_base.cc"

namespace onnx {
namespace sd {
namespace prompt {

using namespace base;
using namespace amon;
using namespace scheduler;
using namespace Ort;
using namespace detail;

typedef struct TokenizerConfig {
    string tokenizer_dictionary_at = "";        // vocabulary lib <one vocab per line, row treate as index>
    int32_t blank_token_size = 49408;           // blank token generated for unconditional input
    int32_t avail_max_length = 77;              // max token length (include <start> & <end>, so 75 avail)
    int32_t major_hidden_dim = 768;             // out token length
    float txt_attn_increase_factor = 1.1f;
    float txt_attn_decrease_factor = 1 / 1.1f;
} TokenizerConfig;

/**
 * Share the same rules with stable-diffusion-webui
 *
 * Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/modules/prompt_parser.py
 */
class PromptParser {
private:
    typedef std::vector<std::pair<std::string, float>> PromptWeight_map;
    typedef std::map<std::string, int> Token_2_ID_dict;
    typedef std::map<int, std::string> ID_2_Token_dict;
    typedef std::vector<int>   Tokens;
    typedef std::vector<float> Multis;

private:
    TokenizerConfig sd_tokenizer_config;
    Token_2_ID_dict sd_tokenizer_tok2id;
    ID_2_Token_dict sd_tokenizer_id2tok;

    const std::regex re_focusing = std::regex(
        R"(\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|\]|[^\\()\[\]:]+|:)"
    );
    const std::regex re_breaking = std::regex(
        R"(\s*\bBREAK\b\s*)"
    );
    const std::regex re_spliting = std::regex(
        (R"(\s+|,)")
    );

private:
    /**
     * @details Method for checking parse_prompt_attention result
     * @param prompt_weight_ split prompt(key_word)-weights map
     */
    void print_prompt_attention(const PromptWeight_map& prompt_weight_) {
        std::stringstream ss;
        ss << "print with format = [[<prompt-key>, <prompt-weight>], ...] below:\n";
        ss << "[";
        for (const auto &item: prompt_weight_) {
            ss << "['" << item.first << "', " << item.second << "], ";
        }
        ss << "]";
        sd_log(LOGGER_INFO) << ss.str().c_str();
    }

    /**
     * @details Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
     * Accepted tokens are:
     *   (abc) - increases attention to abc by a multiplier of 1.1
     *   (abc:3.12) - increases attention to abc by a multiplier of 3.12
     *   [abc] - decreases attention to abc by a multiplier of 1.1
     *   \( - literal character '('
     *   \[ - literal character '['
     *   \) - literal character ')'
     *   \] - literal character ']'
     *   \\ - literal character '\'
     *   anything else - just text
     *
     * @example
     * >>> parse_prompt_attention('normal text')
     * [['normal text', 1.0]]
     * >>> parse_prompt_attention('an (important) word')
     * [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
     * >>> parse_prompt_attention('(unbalanced')
     * [['unbalanced', 1.1]]
     * >>> parse_prompt_attention('\(literal\]')
     * [['(literal]', 1.0]]
     * >>> parse_prompt_attention('(unnecessary)(parens)')
     * [['unnecessaryparens', 1.1]]
     * >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
     * [['a ', 1.0],
     *  ['house', 1.5730000000000004],
     *  [' ', 1.1],
     *  ['on', 1.0],
     *  [' a ', 1.1],
     *  ['hill', 0.55],
     *  [', sun, ', 1.1],
     *  ['sky', 1.4641000000000006],
     *  ['.', 1.1]]
     * @param prompts_ input original prompts with request format<above>
     * @return split prompt(key_word)-weights map
     */
    PromptWeight_map parse_prompt_attention(const std::string& prompts_) {
        PromptWeight_map prompt_weight_;
        std::vector<int> increase_list_;
        std::vector<int> decrease_list_;

        auto multiply_range = [&](int start_position, float multiplier) {
            for (int p = start_position; p < prompt_weight_.size(); ++p) {
                prompt_weight_[p].second *= multiplier;
            }
        };

        std::smatch regex_matcher_;
        std::smatch regex_breaker_;
        std::string remaining_text = prompts_;

        while (std::regex_search(prompts_, regex_matcher_, re_focusing)) {
            std::string text   = regex_matcher_[0];
            std::string weight = regex_matcher_[1];

            if (text == "(") {
                increase_list_.push_back((int)prompt_weight_.size());
            } else if (text == "[") {
                decrease_list_.push_back((int)prompt_weight_.size());
            } else if (!weight.empty() && !increase_list_.empty()) {
                multiply_range(increase_list_.back(), std::stof(weight));
                increase_list_.pop_back();
            } else if (text == ")" && !increase_list_.empty()) {
                multiply_range(increase_list_.back(), sd_tokenizer_config.txt_attn_increase_factor);
                increase_list_.pop_back();
            } else if (text == "]" && !decrease_list_.empty()) {
                multiply_range(decrease_list_.back(), sd_tokenizer_config.txt_attn_decrease_factor);
                decrease_list_.pop_back();
            } else {
                std::vector<std::string> parts = PromptsHelper::split(text, re_breaking);
                for (int i = 0; i < parts.size(); ++i) {
                    if (i > 0) { prompt_weight_.emplace_back("BREAK", -1); }
                    prompt_weight_.emplace_back(parts[i], 1.0f);
                }
            }

            remaining_text = regex_matcher_.suffix();
        }

        for (int pos : increase_list_) {
            multiply_range(pos, sd_tokenizer_config.txt_attn_increase_factor);
        }
        for (int pos : decrease_list_) {
            multiply_range(pos, sd_tokenizer_config.txt_attn_decrease_factor);
        }

        // shrinking results to keep only useful key-weight pair
        if (prompt_weight_.empty()) {
            prompt_weight_.emplace_back("", 1.0f);
        }
        int i = 0;
        while (i + 1 < prompt_weight_.size()) {
            if (prompt_weight_[i].second == prompt_weight_[i + 1].second) {
                prompt_weight_[i].first += prompt_weight_[i + 1].first;
                prompt_weight_.erase(prompt_weight_.begin() + i + 1);
            } else {
                ++i;
            }
        }

        return prompt_weight_;
    }

    std::pair<std::vector<int>, std::vector<float>> parse_attent_tokenized(
        PromptWeight_map prompt_weight_) {

        Tokens remade_tokens;
        Multis multipliers;

        int mark_clip_end_ = 49407;
        int mark_token_id_ = 267;
        int safe_paddings_ = 20;
        int last_comma = -1;

        for (auto concise_: prompt_weight_) {
            std::vector<std::string> words = PromptsHelper::split(concise_.first, re_spliting);
            float weight = concise_.second;

            for (std::string word: words) {
                int token_id_ = sd_tokenizer_tok2id[word];
                bool reach_mark_token_ = (token_id_ == mark_token_id_);
                bool reach_clip_token_ = ((remade_tokens.size() % 75 == 0) && (last_comma != -1) &&
                                          (remade_tokens.size() - last_comma <= safe_paddings_));
                if (reach_mark_token_) {
                    last_comma = remade_tokens.size();
                } else if (reach_clip_token_) {
                    last_comma += 1;
                    Tokens tokens_cache_(remade_tokens.begin() + last_comma, remade_tokens.end());
                    Multis multis_cache_(multipliers.begin() + last_comma, multipliers.end());

                    remade_tokens.resize(last_comma);
                    multipliers.resize(last_comma);

                    int clip_end_ = ceil(remade_tokens.size() / 75.0) * 75 - remade_tokens.size();
                    {
                        remade_tokens.insert(remade_tokens.end(), clip_end_, mark_clip_end_);
                        remade_tokens.insert(remade_tokens.end(), tokens_cache_.begin(), tokens_cache_.end());
                        multipliers.insert(multipliers.end(), clip_end_, 1.0f);
                        multipliers.insert(multipliers.end(), multis_cache_.begin(), multis_cache_.end());
                    }
                }
                remade_tokens.push_back(token_id_);
                multipliers.push_back(weight);
            }
        }

        int prompt_target_length = ceil(remade_tokens.size() / 75.0) * 75;
        int tokens_to_add = prompt_target_length - remade_tokens.size();
        remade_tokens.insert(remade_tokens.end(), tokens_to_add, 49407);
        multipliers.insert(multipliers.end(), tokens_to_add, 1.0f);

        return {remade_tokens, multipliers};
    }

public:
    explicit PromptParser(TokenizerConfig vae_config_ = {}) : sd_tokenizer_config(vae_config_) {};
    ~PromptParser() { sd_tokenizer_config.~TokenizerConfig();};

    void init();
    Tensor tokenize(const std::string& prompts_);
};

void PromptParser::init(){
    std::ifstream vocab_file;
    vocab_file.open(sd_tokenizer_config.tokenizer_dictionary_at.data());
    std::string vocab;
    int idx = 0;
    while (getline(vocab_file, vocab)) {
        sd_tokenizer_tok2id.insert(std::pair<std::string, int>(vocab, idx));
        sd_tokenizer_id2tok.insert(std::pair<int, std::string>(idx, vocab));
        idx++;
    }
    vocab_file.close();
}

std::pair<std::vector<int>, std::vector<float>> PromptParser::remap_tokens(
    Tokens tokens_, Weight weight_
) {
    std::vector<int> remade_tokens;
    std::vector<float> multipliers;

    int last_comma = -1;
    for (int it_tokenized = 0; it_tokenized < tokenized.size(); it_tokenized++) {

        for (int token: tokens_) {
            if (token == 267) {
                last_comma = remade_tokens.size();
            } else if ((remade_tokens.size() % 75 == 0) && (last_comma != -1) &&
                       (remade_tokens.size() - last_comma <= 20)) {
                last_comma += 1;
                std::vector<int> reloc_tokens(remade_tokens.begin() + last_comma, remade_tokens.end());
                std::vector<float> reloc_mults(multipliers.begin() + last_comma, multipliers.end());
                remade_tokens.resize(last_comma);

                int length = remade_tokens.size();
                int rem = ceil(length / 75.0) * 75 - length;
                remade_tokens.insert(remade_tokens.end(), rem, 49407);
                remade_tokens.insert(remade_tokens.end(), reloc_tokens.begin(), reloc_tokens.end());
                multipliers.resize(last_comma);
                multipliers.insert(multipliers.end(), rem, 1.0f);
                multipliers.insert(multipliers.end(), reloc_mults.begin(), reloc_mults.end());
            }
            remade_tokens.push_back(token);
            multipliers.push_back(weight_);
        }
    }

    int prompt_length_ = ceil(remade_tokens.size() / 75.0) * 75;
    int tokens_to_add_ = prompt_length_ - remade_tokens.size();
    remade_tokens.insert(remade_tokens.end(), tokens_to_add_, 49407);
    multipliers.insert(multipliers.end(), tokens_to_add_, 1.0f);

    return {remade_tokens, multipliers};
}

Tensor PromptParser::tokenize(const std::string& prompts_) {

    PromptWeight_map parsed_attention = parse_prompt_attention(prompts_);

    // convert to token_ids(index), careful, the tokenized result is without weight
    std::vector<std::vector<int>> tokenized;
    for (auto concise_: parsed_attention) {
        std::vector<int> tokens;
        std::vector<std::string> words = PromptsHelper::split(concise_.first, re_spliting);
        for (std::string word: words) {
            tokens.push_back(sd_tokenizer_tok2id[word]);
        }
        tokenized.push_back(tokens);
    }

    std::vector<int> remade_tokens;
    std::vector<float> multipliers;
    for (auto concise_: parsed_attention) {
        Tokens tokens;
        Weight weight = concise_.second;
        {
            std::vector<std::string> words = PromptsHelper::split(concise_.first, re_spliting);
            for (std::string word: words) {
                tokens.push_back(sd_tokenizer_tok2id[word]);
            }
        }

        int last_comma = -1;
        for (int i = 0; i < tokens.size(); ++i) {
            int token = tokens[i];
            if (token == 267) {
                last_comma = remade_tokens.size();
            } else if ((max(int(remade_tokens.size()), 1) % 75 == 0) && (last_comma != -1) &&
                       (remade_tokens.size() - last_comma <= 20)) {
                last_comma += 1;
                std::vector<int> reloc_tokens(remade_tokens.begin() + last_comma, remade_tokens.end());
                std::vector<float> reloc_mults(multipliers.begin() + last_comma, multipliers.end());
                std::vector<int> _remade_tokens_(remade_tokens.begin(), remade_tokens.begin() + last_comma);
                remade_tokens = _remade_tokens_;

                int length = remade_tokens.size();
                int rem = ceil(length / 75.0) * 75 - length;
                std::vector<int> tmp_token(rem, 49407);
                remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
                remade_tokens.insert(remade_tokens.end(), reloc_tokens.begin(), reloc_tokens.end());

                std::vector<float> _multipliers_(multipliers.begin(), multipliers.end() + last_comma);
                std::vector<int> tmp_multipliers(rem, 1.0f);
                _multipliers_.insert(_multipliers_.end(), tmp_multipliers.begin(), tmp_multipliers.end());
                _multipliers_.insert(_multipliers_.end(), reloc_mults.begin(), reloc_mults.end());
                multipliers = _multipliers_;
            }
            remade_tokens.push_back(token);
            multipliers.push_back(weight);
        }
    }
    {
        int last_comma = -1;
        for (int it_tokenized = 0; it_tokenized < tokenized.size(); it_tokenized++) {
            Tokens tokens = tokenized[it_tokenized];
            Weight weight = parsed_attention[it_tokenized].second;

            int i = 0;
            while (i < tokens.size()) {
                int token = tokens[i];
                if (token == 267) {
                    last_comma = remade_tokens.size();
                } else if ((max(int(remade_tokens.size()), 1) % 75 == 0) && (last_comma != -1) &&
                           (remade_tokens.size() - last_comma <= 20)) {
                    last_comma += 1;
                    std::vector<int> reloc_tokens(remade_tokens.begin() + last_comma, remade_tokens.end());
                    std::vector<float> reloc_mults(multipliers.begin() + last_comma, multipliers.end());
                    std::vector<int> _remade_tokens_(remade_tokens.begin(), remade_tokens.begin() + last_comma);
                    remade_tokens = _remade_tokens_;
                    int length = remade_tokens.size();
                    int rem = ceil(length / 75.0) * 75 - length;
                    std::vector<int> tmp_token(rem, 49407);
                    remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
                    remade_tokens.insert(remade_tokens.end(), reloc_tokens.begin(), reloc_tokens.end());
                    std::vector<float> _multipliers_(multipliers.begin(), multipliers.end() + last_comma);
                    std::vector<int> tmp_multipliers(rem, 1.0f);
                    _multipliers_.insert(_multipliers_.end(), tmp_multipliers.begin(), tmp_multipliers.end());
                    _multipliers_.insert(_multipliers_.end(), reloc_mults.begin(), reloc_mults.end());
                    multipliers = _multipliers_;
                }
                remade_tokens.push_back(token);
                multipliers.push_back(weight);
                i += 1;
            }
        }
        int prompt_target_length = ceil(max(int(remade_tokens.size()), 1) / 75.0) * 75;
        int tokens_to_add = prompt_target_length - remade_tokens.size();
        std::vector<int> tmp_token(tokens_to_add, 49407);
        remade_tokens.insert(remade_tokens.end(), tmp_token.begin(), tmp_token.end());
        std::vector<int> tmp_multipliers(tokens_to_add, 1.0f);
        multipliers.insert(multipliers.end(), tmp_multipliers.begin(), tmp_multipliers.end());
    }


    std::vector<int> tokens;
    std::vector<float> weights;
    for (const auto& prompt_weight_pair_ : parsed_attention) {
        const std::string& curr_text = prompt_weight_pair_.first;
        float curr_weight            = prompt_weight_pair_.second;
        std::vector<int> curr_tokens = tokenizer->encode(curr_text, on_new_token_cb);
        tokens.insert(tokens.end(), curr_tokens.begin(), curr_tokens.end());
        weights.insert(weights.end(), curr_tokens.size(), curr_weight);
    }

    pad_tokens(tokens, weights, max_length, padding);

    // for (int i = 0; i < tokens.size(); i++) {
    //     std::cout << tokens[i] << ":" << weights[i] << ", ";
    // }
    // std::cout << std::endl;

    return {tokens, weights};
}


} // namespace prompt
} // namespace sd
} // namespace onnx

#endif //MODEL_CLIP_H

