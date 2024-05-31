/*
 * Copyright (c) 2018-2050 SD_Tokenizer(No-model) - Arikan.Li
 *
 * Ref: https://huggingface.co/docs/transformers/tokenizer_summary
 * Created by Arikan.Li on 2024/05/14.
 */
#ifndef TOKENIZER_BASE_H
#define TOKENIZER_BASE_H

#include "onnxsd_foundation.cc"
#include "json.hpp"

namespace onnx {
namespace sd {
namespace tokenizer {

using namespace base;
using namespace amon;

/**
 * Share the same rules with stable-diffusion-webui
 *
 * Ref: https://github.com/AUTOMATIC1111/stable-diffusion-webui/modules/prompt_parser.py
 */
class TokenizerBase {
public:
    typedef std::vector<std::pair<std::string, float>> PromptWeight_map;
    typedef std::vector<std::pair<Tensor, Tensor>> PreparedToken_vec;
    typedef std::vector<std::vector<float>> Embeddings_matrix;
    typedef std::vector<std::vector<float>> Positional_matrix;

protected:
    typedef std::map<std::string, int> Token_2_ID_dict;
    typedef std::map<int, std::string> ID_2_Token_dict;
    typedef std::vector<int32_t> Tokens;
    typedef std::vector<float> Multis;

protected:
    TokenizerConfig sd_tokenizer_config;
    Token_2_ID_dict sd_tokenizer_tok2id;
    ID_2_Token_dict sd_tokenizer_id2tok;
    Embeddings_matrix embeddings_matrix;
    Positional_matrix positional_matrix;

    const std::regex re_focusing = std::regex(
        R"(\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|:([+-]?[.\d]+)\)|\)|\]|[^\\()\[\]:]+|:)"
    );
    const std::regex re_breaking = std::regex(
        R"(\s*\bBREAK\b\s*)"
    );

protected:
    /**
     * @details Method for checking parse_prompt_attention result
     * @param prompt_weight_ split prompt(key_word)-weights map
     */
    void print_prompt_attention(const PromptWeight_map &prompt_weight_) const {
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
     * @details Check tensor_ range, to prevent idx error setting. Method for debugging
     * @param tensor_ to check
     * @param min_value_ min limit
     * @param max_value_ max limit
     * @return if out of range, then return true
     */
    bool check_tensor_range(const Tensor& tensor_, int64_t min_value_, int64_t max_value_) {
        bool match_case_ = false;
        auto tensor_info = tensor_.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        size_t total_elements = tensor_info.GetElementCount();

        auto data = tensor_.GetTensorData<int32_t>();
        for (size_t i = 0; i < total_elements; ++i) {
            if (data[i] < min_value_ || data[i] > max_value_) {
                std::cerr << "Error: Index " << data[i]
                          << " at position " << i << " is out of bounds. Must be within the range ["
                          << min_value_ << ", " << max_value_ << "]." << std::endl;
                match_case_ = true;
            }
        }
        return match_case_;
    }

    /**
     * @details get available customizable token max size, set by config[avail_token_size]
     * @return always equal to <avail_token_size - 2>
     */
    int32_t get_avail_token_size() const {
        return sd_tokenizer_config.avail_token_size - 2;
    }

    /**
     * @details get <|startoftext|> index in dictionary, set by config[tokenizer_dictionary_at]
     * @return <|startoftext|> index in dictionary
     */
    int32_t get_start_token_index() const {
        return sd_tokenizer_config.avail_token_count - 2;
    }

    /**
     * @details get <|endoftext|> index in dictionary, set by config[tokenizer_dictionary_at]
     * @return <|endoftext|> index in dictionary
     */
    int32_t get_end_token_index() const {
        return sd_tokenizer_config.avail_token_count - 1;
    }

    /**
     * @details get token index for padding, its different defined by mode
     * Examples:
     *    SD - Using <|endoftext|> = 49407 as pad_token_idx
     *    SDXL - Using simple Nan-Mark = 0 as pad_token_idx
     * @return padding token index in model setting
     */
    int32_t get_pad_token_index() const {
        return get_end_token_index();
    }

    /**
     * @details get <|startoftext|>  <|endoftext|> etc. marks multiply,
     * set by config[major_boundary_factor]
     * @return <|startoftext|> index in dictionary
     */
    float get_boundary_factor() const {
        return sd_tokenizer_config.major_boundary_factor;
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
     * >>> parse_prompt_attention('normal text') \n
     * [['normal text', 1.0]] \n
     * >>> parse_prompt_attention('an (important) word') \n
     * [['an ', 1.0], ['important', 1.1], [' word', 1.0]] \n
     * >>> parse_prompt_attention('(unbalanced') \n
     * [['unbalanced', 1.1]] \n
     * >>> parse_prompt_attention('\(literal\]') \n
     * [['(literal]', 1.0]] \n
     * >>> parse_prompt_attention('(unnecessary)(parens)') \n
     * [['unnecessaryparens', 1.1]] \n
     * >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).') \n
     * [['a ', 1.0],
     *  ['house', 1.5730000000000004],
     *  [' ', 1.1],
     *  ['on', 1.0],
     *  [' a ', 1.1],
     *  ['hill', 0.55],
     *  [', sun, ', 1.1],
     *  ['sky', 1.4641000000000006],
     *  ['.', 1.1]] \n
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

        while (std::regex_search(remaining_text, regex_matcher_, re_focusing)) {
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

    void load_vocab_json(const std::string &vocab_path_) {
        std::ifstream vocab_file(vocab_path_);
        if (!vocab_file) {
            std::cerr << "Failed to open " << vocab_path_ << std::endl;
            exit(1);
        }

        nlohmann::json json;
        vocab_file >> json;

        for (auto it = json.begin(); it != json.end(); ++it) {
            std::string str_key = it.key();
            int int_idx = it.value().get<int>();

            str_key = PromptsHelper::replace(str_key, "\\u0120", " ");  // \u0120 -> space
            str_key = PromptsHelper::replace(str_key, "\\u010a", "\n"); // \u010a -> new line
            str_key = PromptsHelper::replace(str_key, "\\\"", "\"");    // \\\"   -> "

            sd_tokenizer_tok2id[str_key] = int_idx;
            sd_tokenizer_id2tok[int_idx] = str_key;
        }
    }

    void load_vocab_text(const std::string & vocab_path_) {
        std::ifstream vocab_file;
        vocab_file.open(vocab_path_);
        std::string vocab;
        int idx = 0;
        while (getline(vocab_file, vocab)) {
            sd_tokenizer_tok2id.insert(std::pair<std::string, int>(vocab, idx));
            sd_tokenizer_id2tok.insert(std::pair<int, std::string>(idx, vocab));
            idx++;
        }
        vocab_file.close();
    }

private:        // WARNING: Test ONLY!
    void test_encoder_prepare() {
        // test(simple encoding): prepare token embeddings_matrix
        {
            embeddings_matrix.resize(
                sd_tokenizer_config.avail_token_count,
                vector<float>(sd_tokenizer_config.major_hidden_dim)
            );
            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<> dis(-0.1, 0.1);

            for (int i = 0; i < sd_tokenizer_config.avail_token_count; ++i) {
                for (int j = 0; j < sd_tokenizer_config.major_hidden_dim; ++j) {
                    embeddings_matrix[i][j] = dis(gen);
                }
            }
        }

        // test(simple encoding): prepare token positional_matrix
        {
            positional_matrix.resize(
                sd_tokenizer_config.avail_token_size,
                vector<float>(sd_tokenizer_config.major_hidden_dim, 0.0)
            );

            for (int pos = 0; pos < sd_tokenizer_config.avail_token_size; ++pos) {
                for (int i = 0; i < sd_tokenizer_config.major_hidden_dim; i += 2) {
                    positional_matrix[pos][i] = sin(
                        pos / pow(10000, 2.0 * i / sd_tokenizer_config.major_hidden_dim));
                    if (i + 1 < sd_tokenizer_config.major_hidden_dim) {
                        positional_matrix[pos][i + 1] = cos(
                            pos / pow(10000, 2.0 * (i + 1) / sd_tokenizer_config.major_hidden_dim));
                    }
                }
            }
        }
    }

    Tensor test_encoding(const Tensor &token_) {
        auto *token_data_ = token_.GetTensorData<int>();
        long token_size_ = TensorHelper::get_data_size(token_);

        std::vector<float> result_(token_size_ * sd_tokenizer_config.major_hidden_dim);

        for (int i = 0; i < sd_tokenizer_config.avail_token_size; ++i) {
            std::vector<float> embeddings_ = embeddings_matrix[token_data_[i]];
            std::vector<float> positional_ = positional_matrix[i];

            int offset_ = sd_tokenizer_config.avail_token_size * i;
            for (int j = 0; j < sd_tokenizer_config.major_hidden_dim; ++j) {
                result_[offset_ + j] = embeddings_[j] + positional_[j];
            }
        }

        TensorShape shape_ = {1, sd_tokenizer_config.avail_token_size, sd_tokenizer_config.major_hidden_dim};
        return TensorHelper::create(shape_, result_);
    }

protected:
    virtual std::tuple<Tokens, Multis, size_t> encode(PromptWeight_map prompt_weight_) = 0;

public:
    explicit TokenizerBase(const TokenizerConfig &config_ = DEFAULT_TOKENIZER_CONFIG) : sd_tokenizer_config(config_) {};
    virtual ~TokenizerBase() { sd_tokenizer_config.~TokenizerConfig();};

    void create();
    void init();
    PreparedToken_vec tokenize(const std::string &prompts_);
    Tensor embedding(const Tensor &token_p_,const Tensor &token_n_);
    std::string untokenize(const std::pair<Tensor, Tensor> &tpair_);
    void uninit();
    void release();
};

void TokenizerBase::create() {
}

void TokenizerBase::init(){
    // loading vocabulary
    if (PromptsHelper::has_extension(sd_tokenizer_config.tokenizer_dictionary_at, ".json")) {
        load_vocab_json(sd_tokenizer_config.tokenizer_dictionary_at);
    } else if (PromptsHelper::has_extension(sd_tokenizer_config.tokenizer_dictionary_at, ".txt")) {
        load_vocab_text(sd_tokenizer_config.tokenizer_dictionary_at);
    }
    test_encoder_prepare();
}

TokenizerBase::PreparedToken_vec TokenizerBase::tokenize(const std::string& prompts_) {

    PreparedToken_vec matched_results_;

    bool need_uncond_manual = prompts_.empty();
    if (need_uncond_manual) {
        // manual make uncondtational <token_idx, weight>
        TensorShape paired_shape_ = {1, sd_tokenizer_config.avail_token_size};
        Tokens tokens_cache_(sd_tokenizer_config.avail_token_size);
        Multis multis_cache_(sd_tokenizer_config.avail_token_size);
        {
            tokens_cache_[0] = get_start_token_index();
            tokens_cache_[1] = get_end_token_index();
            multis_cache_[0] = get_boundary_factor();
            multis_cache_[1] = get_boundary_factor();
            for (int i = 2; i < sd_tokenizer_config.avail_token_size; ++i) {
                tokens_cache_[i] = get_pad_token_index();
                multis_cache_[i] = get_boundary_factor();
            }
        }
        Tensor token_tensor = TensorHelper::create<int32_t>(paired_shape_, tokens_cache_);
        Tensor multi_tensor = TensorHelper::create<float>(paired_shape_, multis_cache_);
        matched_results_.emplace_back(std::move(token_tensor), std::move(multi_tensor));
    } else {
        // parsing input prompts to get <token_idx, weight>
        PromptWeight_map cur_parsed_attention = parse_prompt_attention(prompts_);
        std::tuple<Tokens, Multis, size_t> encoded_input = encode(cur_parsed_attention);     // {tokens, weights}

        Tokens encoded_tokens_ = std::get<0>(encoded_input);
        Multis encoded_multis_ = std::get<1>(encoded_input);
        size_t encoded_pair_num = std::get<2>(encoded_input);
        const int avail_token_size_ = get_avail_token_size();      // limit of current token_size

        auto encoded_token_index_ = encoded_tokens_.begin();
        auto encoded_multi_index_ = encoded_multis_.begin();
        for (int i = 0; i < encoded_pair_num; ++i) {
            Tokens tokens_cache_(encoded_token_index_, encoded_token_index_ + avail_token_size_);
            Multis multis_cache_(encoded_multi_index_, encoded_multi_index_ + avail_token_size_);
            tokens_cache_.insert(tokens_cache_.begin(), get_start_token_index());
            multis_cache_.insert(multis_cache_.begin(), get_boundary_factor());
            tokens_cache_.push_back(get_end_token_index());
            multis_cache_.push_back(get_boundary_factor());

            TensorShape paired_shape_ = {1, sd_tokenizer_config.avail_token_size};
            Tensor token_tensor = TensorHelper::create<int32_t>(paired_shape_, tokens_cache_);
            Tensor multi_tensor = TensorHelper::create<float>(paired_shape_, multis_cache_);
            matched_results_.emplace_back(std::move(token_tensor), std::move(multi_tensor));

            check_tensor_range(matched_results_.back().first, -49408, 49407);

            encoded_token_index_= encoded_token_index_ + avail_token_size_;
            encoded_multi_index_= encoded_multi_index_ + avail_token_size_;
        }
    }

    return matched_results_;
}

Tensor TokenizerBase::embedding(const Tensor &token_p_, const Tensor &token_n_) {
    Tensor encoded_token_ = TensorHelper::clone<float>(token_p_);
    Tensor unconditional_ = TensorHelper::clone<float>(token_n_);

    // TODO Must manual match, if two input channel is different

    std::vector<Tensor> target_tensors;
    target_tensors.emplace_back(std::move(encoded_token_));
    target_tensors.emplace_back(std::move(unconditional_));
    return TensorHelper::merge(target_tensors, 0);
}

std::string TokenizerBase::untokenize(const  std::pair<Tensor, Tensor>& tpair_) {

    // TODO PromptWeight_map parsed_attention = decode(embeds_);

    // TODO print_prompt_attention(parsed_attention);

    // in current situations seems unnecessary

    return "";
}

void TokenizerBase::uninit() {
}

void TokenizerBase::release() {
}

} // namespace tokenizer
} // namespace sd
} // namespace onnx

#endif //TOKENIZER_BASE_H

