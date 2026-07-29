/*
 * Copyright (c) 2018-2050 SD_Scheduler
 * Created by Arikan.Li on 2024/07/16.
 * Completed & verified on 2026/07/29.
 *
 * Unified Predictor-Corrector Method (UniPC)
 * base on: https://arxiv.org/pdf/2302.04867
 *
 * Derivation notes (x0-parameterization):
 *   samples in this codebase live in EDM convention x = x0 + σ·ε with σ = σ_ratio
 *   (see SchedulerBase::scale / find_predict_params_at, α≡1 in sample space).
 *   the marginal-consistent ODE solution in λ = -ln(σ) is:
 *       x_t = (σ_t/σ_s) * x_s + ∫_{λs}^{λt} e^{λ-λt} * x0(λ) dλ
 *   verified against constant-epsilon (euler/DDIM) and constant-x0 limits.
 *   x0(λ) is Lagrange-interpolated over [m0(current), m1..m_{p-1}(history)],
 *   basis integrals are evaluated in the e^{-h}-normalized (overflow-free) form.
 */
#ifndef SCHEDULER_DISCRETE_UNIPC
#define SCHEDULER_DISCRETE_UNIPC

#include "scheduler_base.cc"

namespace onnx {
namespace sd {
namespace scheduler {

class UniPCDiscreteScheduler: public SchedulerBase {
private:
    typedef std::vector<float> UniData;
    typedef std::vector<double> CoefData;

    static constexpr float  SD_LAMBDA_FLOOR_SIGMA = 1e-7f;   // sigma floor for lambda evaluation
    static constexpr double SD_LAMBDA_FLOOR_H     = 1e-12;   // h floor to keep recurrences sane
    static constexpr double SD_LAMBDA_JUMP_CAP    = 1.25;    // drop to order-1 beyond this λ-jump
                                                             // (λ-extrapolation stability guard,
                                                             //  generalizes diffusers lower_order_final)

    std::vector<UniData> history_dnoise;       // model x0-predictions, newest first
    std::vector<double>  history_lambda;       // lambda value of each history entry, newest first
    UniData              last_samples_;        // sample produced at previous step

private:
    /* numeric assistants ===================================================*/
    static double lambda_at(float sigma_);                       // λ = -ln(σ_ratio), sigma floored

    long   get_unified_history_count(long step_index_) const;
    UniData get_unified_correction(const UniData& curs_dnoised_, long step_index_);
    UniData get_unified_prediction(const UniData& curs_samples_, long step_index_);

    // expand Lagrange basis L_k(ξ) (points r_[] with r_[0]=0) into power coefficients
    static std::vector<CoefData> lagrange_power_coefs(const CoefData& r_);
    // Ã_k = ∫_0^h  e^{ξ-h} L_k(ξ) dξ   (UniP kernel, stable for any h >= 0)
    static CoefData integrate_basis_predictor(const CoefData& r_, double h_);
    // Ã_k = ∫_a^0  e^{ξ}   L_k(ξ) dξ   (UniC kernel, a < 0 finite interval)
    static CoefData integrate_basis_corrector(const CoefData& r_, double a_);

protected:
    std::vector<float> execute_method(
        const float* predict_data_,
        const float* samples_data_,
        long data_size_,
        long step_index_,
        float random_intensity_
    ) override;

public:
    explicit UniPCDiscreteScheduler(SchedulerConfig scheduler_config_ = {}) : SchedulerBase(scheduler_config_) {
    }

    ~UniPCDiscreteScheduler() override = default;
};

/* Assistant Operations ===================================================*/

double UniPCDiscreteScheduler::lambda_at(float sigma_) {
    return -std::log(double(std::max(sigma_, SD_LAMBDA_FLOOR_SIGMA)));
}

long UniPCDiscreteScheduler::get_unified_history_count(long step_index_) const {
    long maintain_order_ = long(scheduler_config.scheduler_maintain_cache);
    return std::min(maintain_order_, step_index_);
}

// L_k(ξ) = Π_{j≠k} (ξ - r_j) / (r_k - r_j), returned as power-series coefficients
std::vector<UniPCDiscreteScheduler::CoefData> UniPCDiscreteScheduler::lagrange_power_coefs(
    const CoefData& r_
) {
    size_t p = r_.size();
    std::vector<CoefData> result(p, CoefData(p, 0.0));
    for (size_t k = 0; k < p; k++) {
        CoefData poly(1, 1.0);                     // running polynomial Π(ξ - r_j)
        double denom = 1.0;
        for (size_t j = 0; j < p; j++) {
            if (j == k) continue;
            CoefData next(poly.size() + 1, 0.0);   // multiply by (ξ - r_j)
            for (size_t i = 0; i < poly.size(); i++) {
                next[i + 1] += poly[i];            // * ξ
                next[i]     -= poly[i] * r_[j];    // * (-r_j)
            }
            poly.swap(next);
            denom *= (r_[k] - r_[j]);
        }
        for (size_t i = 0; i < p; i++) {
            result[k][i] = poly[i] / denom;
        }
    }
    return result;
}

// J_i(h) = ∫_0^h ξ^i e^{ξ-h} dξ  via  J_0 = 1-e^{-h}, J_i = h^i - i*J_{i-1}
UniPCDiscreteScheduler::CoefData UniPCDiscreteScheduler::integrate_basis_predictor(
    const CoefData& r_, double h_
) {
    size_t p = r_.size();
    h_ = std::max(h_, SD_LAMBDA_FLOOR_H);

    CoefData J(p, 0.0);
    J[0] = 1.0 - std::exp(-h_);
    double h_pow = h_;
    for (size_t i = 1; i < p; i++) {
        J[i] = h_pow - double(i) * J[i - 1];
        h_pow *= h_;
    }

    std::vector<CoefData> basis = lagrange_power_coefs(r_);
    CoefData coefs(p, 0.0);
    for (size_t k = 0; k < p; k++) {
        for (size_t i = 0; i < p; i++) {
            coefs[k] += basis[k][i] * J[i];
        }
    }
    return coefs;
}

// F_i(x) = ∫ ξ^i e^ξ dξ = e^x P_i(x), P_0=1, P_i = x^i - i*P_{i-1}; ∫_a^0 = P_i(0) - e^a P_i(a)
UniPCDiscreteScheduler::CoefData UniPCDiscreteScheduler::integrate_basis_corrector(
    const CoefData& r_, double a_
) {
    size_t p = r_.size();

    CoefData P0(p, 0.0), Pa(p, 0.0);               // P_i(0) and P_i(a)
    P0[0] = 1.0;
    Pa[0] = 1.0;
    double a_pow = a_;
    for (size_t i = 1; i < p; i++) {
        P0[i] = -double(i) * P0[i - 1];
        Pa[i] = a_pow - double(i) * Pa[i - 1];
        a_pow *= a_;
    }
    double exp_a = std::exp(a_);

    std::vector<CoefData> basis = lagrange_power_coefs(r_);
    CoefData coefs(p, 0.0);
    for (size_t k = 0; k < p; k++) {
        for (size_t i = 0; i < p; i++) {
            coefs[k] += basis[k][i] * (P0[i] - exp_a * Pa[i]);
        }
    }
    return coefs;
}

/* Essential Operations ===================================================*/

UniPCDiscreteScheduler::UniData UniPCDiscreteScheduler::get_unified_correction(
    const UniData& curs_dnoised_, long step_index_
) {
    size_t data_size_ = curs_dnoised_.size();
    float sigma_curs = scheduler_sigmas[size_t(step_index_)];
    float sigma_prev = scheduler_sigmas[size_t(step_index_ - 1)];

    double lambda_s0 = lambda_at(sigma_curs);
    double a_        = lambda_at(sigma_prev) - lambda_s0;          // < 0

    // interpolation points: m0(current, r=0) + history, bounded by maintain order;
    // drop to order-1 when the correction interval is a huge λ-jump (see SD_LAMBDA_JUMP_CAP)
    long order_ = std::min<long>(long(history_dnoise.size()) + 1,
                                 long(scheduler_config.scheduler_maintain_cache));
    if (-a_ > SD_LAMBDA_JUMP_CAP) order_ = 1;
    CoefData r(size_t(order_), 0.0);
    for (long k = 1; k < order_; k++) {
        r[size_t(k)] = history_lambda[size_t(k - 1)] - lambda_s0;
    }
    CoefData coefs = integrate_basis_corrector(r, a_);

    // x_s0 = (σ_s0/σ_prev) * last + Σ Ã_k m_k
    double f_ = double(sigma_curs) / double(sigma_prev);
    UniData corrected_(data_size_, 0.0f);
    for (size_t i = 0; i < data_size_; i++) {
        double accum = 0.0;
        accum += coefs[0] * double(curs_dnoised_[i]);
        for (long k = 1; k < order_; k++) {
            accum += coefs[size_t(k)] * double(history_dnoise[size_t(k - 1)][i]);
        }
        corrected_[i] = float(f_ * double(last_samples_[i]) + accum);
    }
    return corrected_;
}

UniPCDiscreteScheduler::UniData UniPCDiscreteScheduler::get_unified_prediction(
    const UniData& curs_samples_, long step_index_
) {
    size_t data_size_ = curs_samples_.size();
    float sigma_curs = scheduler_sigmas[size_t(step_index_)];
    float sigma_next = scheduler_sigmas[size_t(step_index_ + 1)];  // appended 0 at final step

    // final step targets σ=0: exact limit of the ODE solution is the x0-prediction itself
    if (sigma_next <= SD_LAMBDA_FLOOR_SIGMA) {
        return history_dnoise[0];
    }

    double lambda_s0 = lambda_at(sigma_curs);
    double lambda_t  = lambda_at(sigma_next);
    double h_        = lambda_t - lambda_s0;                       // > 0

    // history already holds m0 at front after update;
    // drop to order-1 when the prediction interval is a huge λ-jump (see SD_LAMBDA_JUMP_CAP)
    long order_ = std::min<long>(long(history_dnoise.size()),
                                 long(scheduler_config.scheduler_maintain_cache));
    if (h_ > SD_LAMBDA_JUMP_CAP) order_ = 1;
    CoefData r(size_t(order_), 0.0);
    for (long k = 1; k < order_; k++) {
        r[size_t(k)] = history_lambda[size_t(k)] - lambda_s0;
    }
    CoefData coefs = integrate_basis_predictor(r, h_);

    // x_t = (σ_t/σ_s0) * x + Σ Ã_k m_k
    double f_ = double(sigma_next) / double(sigma_curs);

    UniData predicted_(data_size_, 0.0f);
    for (size_t i = 0; i < data_size_; i++) {
        double accum = 0.0;
        for (long k = 0; k < order_; k++) {
            accum += coefs[size_t(k)] * double(history_dnoise[size_t(k)][i]);
        }
        predicted_[i] = float(f_ * double(curs_samples_[i]) + accum);
    }
    return predicted_;
}

/**
 * UniPC main step: correct -> record -> predict
 */
std::vector<float> UniPCDiscreteScheduler::execute_method(
    const float* predict_data_,
    const float* samples_data_,
    long data_size_,
    long step_index_,
    float random_intensity_
) {
    SD_UNUSED(random_intensity_);

    // predict_data_ is already the x0-prediction (converted by base with c_skip/c_out)
    UniData curs_dnoised_(predict_data_, predict_data_ + data_size_);
    UniData curs_samples_(samples_data_, samples_data_ + data_size_);
    double lambda_s0 = lambda_at(scheduler_sigmas[size_t(step_index_)]);

    // UniC: correct previous sample with current model output (from the 2nd step on)
    if (step_index_ > 0 && !history_dnoise.empty() && long(last_samples_.size()) == data_size_) {
        curs_samples_ = get_unified_correction(curs_dnoised_, step_index_);
    }

    // UniPC: update history records, insert m0 & λ0 to records->front
    {
        history_dnoise.insert(history_dnoise.begin(), curs_dnoised_);
        history_lambda.insert(history_lambda.begin(), lambda_s0);
        long maintain_order_ = std::max(1L, long(scheduler_config.scheduler_maintain_cache));
        while (long(history_dnoise.size()) > maintain_order_) {
            history_dnoise.pop_back();
            history_lambda.pop_back();
        }
        last_samples_ = curs_samples_;
    }

    // UniP: predict next sample from the corrected current state
    return get_unified_prediction(curs_samples_, step_index_);
}

} // namespace scheduler
} // namespace sd
} // namespace onnx

#endif //SCHEDULER_DISCRETE_UNIPC
