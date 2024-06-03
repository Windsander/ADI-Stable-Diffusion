from PIL import Image
import onnxruntime as ort
import numpy as np
import math


class SchedulerBase:
    def __init__(self, scheduler_config):
        self.scheduler_config = scheduler_config
        self.scheduler_timesteps = {}
        self.scheduler_sigmas = []
        self.alphas_cumprod = []
        self.scheduler_max_sigma = 0
        self.random_generator = np.random.default_rng(scheduler_config['scheduler_seed'])

    def find_closest_timestep_index(self, time_):
        keys = list(self.scheduler_timesteps.keys())
        values = list(self.scheduler_timesteps.values())
        idx = np.searchsorted(values, time_)
        if idx == len(values):
            raise RuntimeError("closest index found failed")
        if idx > 0 and (idx == len(values) or abs(values[idx] - time_) >= abs(values[idx - 1] - time_)):
            idx -= 1
        return keys[idx]

    def generate_sigma_at(self, timestep_):
        low_idx = int(math.floor(timestep_))
        high_idx = int(math.ceil(timestep_))
        l_sigma = math.log(self.alphas_cumprod[low_idx])
        h_sigma = math.log(self.alphas_cumprod[high_idx])
        w = timestep_ - low_idx
        sigma = (1.0 - w) * l_sigma + w * h_sigma
        return math.exp(sigma)

    def create(self):
        training_steps = self.scheduler_config['scheduler_training_steps']
        linear_start = self.scheduler_config['scheduler_beta_start']
        linear_end = self.scheduler_config['scheduler_beta_end']
        beta_type = self.scheduler_config['scheduler_beta_type']
        alpha_type = self.scheduler_config['scheduler_alpha_type']

        if beta_type == 'BETA_TYPE_LINEAR':
            beta_start_at = linear_start
            beta_end_when = linear_end
            beta_range = beta_end_when - beta_start_at
            product = 1.0

            for i in range(training_steps):
                beta_norm = beta_start_at + beta_range * (i / (training_steps - 1))
                product *= 1.0 - beta_norm
                comprod_sigma = math.sqrt((1 - product) / product)
                self.alphas_cumprod.append(comprod_sigma)

        elif beta_type == 'BETA_TYPE_SCALED_LINEAR':
            beta_start_at = math.sqrt(linear_start)
            beta_end_when = math.sqrt(linear_end)
            beta_range = beta_end_when - beta_start_at
            product = 1.0

            for i in range(training_steps):
                beta_dire = beta_start_at + beta_range * (i / (training_steps - 1))
                beta_norm = beta_dire ** 2
                product *= 1.0 - beta_norm
                comprod_sigma = math.sqrt((1 - product) / product)
                self.alphas_cumprod.append(comprod_sigma)

        elif beta_type == 'BETA_TYPE_SQUAREDCOS_CAP_V2':
            beta_max = 0.999
            product = 1.0

            def alpha_bar_fn(f_step):
                if alpha_type == 'ALPHA_TYPE_COSINE':
                    return math.cos((f_step + 0.008) / 1.008 * math.pi / 2) ** 2
                elif alpha_type == 'ALPHA_TYPE_EXP':
                    return math.exp(f_step * -12.0)
                else:
                    return 1.0

            for i in range(training_steps):
                t1 = i / training_steps
                t2 = (i + 1) / training_steps
                beta_norm = min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), beta_max)
                product *= 1.0 - beta_norm
                comprod_sigma = math.sqrt((1 - product) / product)
                self.alphas_cumprod.append(comprod_sigma)

        else:
            raise NotImplementedError("ERROR:: PREDICT_TYPE_SAMPLE unimplemented")

    def init(self, inference_steps):
        if inference_steps == 0:
            raise ValueError("ERROR:: inference_steps setting with 0!")

        start_at = 0
        end_when = self.scheduler_config['scheduler_training_steps'] - 1
        step_gap = (float(end_when - start_at) / float(inference_steps - 1)) if inference_steps > 1 else float(end_when)

        for i in range(inference_steps):
            t = float(end_when) - step_gap * float(i)
            sigma = self.generate_sigma_at(t)
            self.scheduler_timesteps[i] = t
            self.scheduler_sigmas.append(sigma)
            self.scheduler_max_sigma = max(self.scheduler_max_sigma, sigma)
        self.scheduler_sigmas.append(0)

    def mask(self, mask_shape):
        return self.random_generator.normal(size=mask_shape) * self.scheduler_max_sigma

    def scale(self, masker, step_index):
        if step_index >= len(self.scheduler_timesteps):
            raise IndexError("from time not found target TimeSteps.")
        sigma = self.scheduler_sigmas[step_index]
        sigma = math.sqrt(sigma * sigma + 1)
        return masker / sigma

    def time(self, step_index):
        if step_index >= len(self.scheduler_timesteps):
            raise IndexError("from time not found target TimeSteps.")
        timestep_value = [self.scheduler_timesteps[step_index]]
        return np.array(timestep_value, dtype=np.int64)

    def step(self, sample, dnoise, step_index, order=4):
        if step_index >= len(self.scheduler_timesteps):
            raise IndexError("from time not found target TimeSteps.")

        output_shape = sample.shape
        data_size = sample.size
        sample_data = sample.flatten()
        dnoise_data = dnoise.flatten()
        predict_data = np.zeros(data_size)

        sigma = self.scheduler_sigmas[step_index]
        for i in range(data_size):
            if self.scheduler_config['scheduler_predict_type'] == 'PREDICT_TYPE_EPSILON':
                predict_data[i] = sample_data[i] - dnoise_data[i] * sigma
            elif self.scheduler_config['scheduler_predict_type'] == 'PREDICT_TYPE_V_PREDICTION':
                predict_data[i] = (sample_data[i] / (sigma ** 2 + 1)) + (
                        dnoise_data[i] * (-sigma / math.sqrt(sigma ** 2 + 1)))
            elif self.scheduler_config['scheduler_predict_type'] == 'PREDICT_TYPE_SAMPLE':
                raise NotImplementedError("ERROR:: PREDICT_TYPE_SAMPLE unimplemented")

        latent_value = self.euler_a_execute_method(predict_data, sample_data, data_size, step_index, order)
        result_latent = np.reshape(latent_value, output_shape)

        return result_latent

    def uninit(self):
        self.scheduler_timesteps.clear()
        self.scheduler_sigmas.clear()

    def release(self):
        self.alphas_cumprod.clear()

    def euler_a_execute_method(self, predict_data, samples_data, data_size, step_index, order):
        eular_a_random = np.random.default_rng()  # random number
        # 忽略 order 参数
        scaled_sample = np.zeros(data_size)

        # Euler method:: sigma get
        sigma_curs = self.scheduler_sigmas[step_index]
        sigma_next = self.scheduler_sigmas[step_index + 1]
        sigma_up = 0
        sigma_dt = 0

        sigma_curs_pow = sigma_curs * sigma_curs
        sigma_next_pow = sigma_next * sigma_next
        sigma_up_numerator = sigma_next_pow * (sigma_curs_pow - sigma_next_pow)
        sigma_down = np.sqrt(sigma_next * sigma_next - sigma_up * sigma_up)
        sigma_up = min(sigma_next, np.sqrt(sigma_up_numerator / sigma_curs_pow))
        sigma_dt = sigma_down - sigma_curs

        # Euler method:: current noise decrees
        for i in range(data_size):
            scaled_sample[i] = (samples_data[i] - predict_data[
                i]) / sigma_curs  # derivative_out = (sample - predict_sample) / sigma
            scaled_sample[i] = samples_data[i] + scaled_sample[
                i] * sigma_dt  # previous_down = sample + derivative_out * dt
            if sigma_next > 0:
                scaled_sample[
                    i] += eular_a_random.normal() * sigma_up  # producted_out = previous_down + random_noise * sigma_up

        return scaled_sample.tolist()


class UNet:
    def __init__(self, sd_unet_config, sd_scheduler, sd_executor):
        self.sd_unet_config = sd_unet_config
        self.sd_scheduler = sd_scheduler
        self.sd_executor = sd_executor
        sd_scheduler.create()
        sd_scheduler.init(sd_unet_config['sd_inference_steps'])
        print(sd_scheduler.alphas_cumprod)
        print(sd_scheduler.scheduler_sigmas)

    def inference(self, embs_positive, embs_negative, encoded_img):
        w = int(self.sd_unet_config['sd_input_width'])
        h = int(self.sd_unet_config['sd_input_height'])
        c = int(self.sd_unet_config['sd_input_channel'])
        need_guidance = (self.sd_unet_config['sd_scale_guidance'] > 1)

        print("embs_positive:", embs_positive)

        latent_shape = (1, c, h, w)
        latent_empty = np.zeros((c * h * w,), dtype=np.float32)
        latents = (encoded_img if encoded_img.size > 0 else latent_empty).reshape(latent_shape)
        init_mask = self.sd_scheduler.mask(latent_shape)
        latents = latents + init_mask

        for i in range(self.sd_unet_config['sd_inference_steps']):
            model_latent = self.sd_scheduler.scale(latents, i).reshape(latent_shape)
            timestep = self.sd_scheduler.time(i)
            print("sample shape:", model_latent.shape)
            print("timestep shape:", timestep.shape)

            # Prepare input tensors
            pred_positive = np.zeros(0)
            if embs_positive.size > 0:
                print("embs_positive shape:", embs_positive.shape)
                output_tensors = self.sd_executor.run(
                    ['out_sample'], {
                        'sample': model_latent.astype(np.float32),
                        'timestep': timestep.astype(np.int64),
                        'encoder_hidden_states': embs_positive.astype(np.float32),
                    }
                )
                pred_positive = output_tensors[0]
                print("pred_positive shape:", pred_positive.shape)

            pred_negative = np.zeros(0)
            if embs_negative.size > 0:
                print("embs_negative shape:", embs_negative.shape)
                output_tensors = self.sd_executor.run(
                    ['out_sample'], {
                        'sample': model_latent.astype(np.float32),
                        'timestep': timestep.astype(np.int64),
                        'encoder_hidden_states': embs_negative.astype(np.float32),
                    }
                )
                pred_negative = output_tensors[0]
                print("pred_negative shape:", pred_negative.shape)

            # Merge predictions
            merge_factor = self.sd_unet_config['sd_scale_guidance']
            guided_pred = (self.guidance(pred_positive, embs_negative, merge_factor) if need_guidance else
                           pred_positive.copy())

            # Dnoise & Step
            latents = self.sd_scheduler.step(latents, guided_pred, i)

            self.print_progress_bar(float(i + 1) / float(self.sd_unet_config['sd_inference_steps']))

        return latents

    def guidance(self, pred_normal, pred_uncond, merge_factor):
        # Placeholder for guidance function
        # This should be implemented based on your model's requirements
        return pred_normal * merge_factor + pred_uncond * (1 - merge_factor)

    def print_progress_bar(self, progress):
        # Placeholder for printing progress bar
        # This should be implemented based on your requirements
        print(f"Progress: {progress * 100:.2f}%")


class Clip:
    def __init__(self, sd_executor):
        self.sd_executor = sd_executor

    def conditional_tokens(self):
        MAX_LENGTH = 77  # Depend on model setting
        BOS_TOKEN = 49406  # Depend on model setting
        PAD_TOKEN = 49407  # Depend on model setting
        output = np.full(MAX_LENGTH, PAD_TOKEN, dtype=np.int32)
        output[0] = BOS_TOKEN
        output[1] = 288  # 'A</w>'
        output[2] = 2368  # 'cat</w>'
        output[3] = 530  # 'in</w>'
        output[4] = 518  # 'the</w>'
        output[5] = 1573  # 'water</w>'
        output[6] = 536  # 'at</w>'
        output[7] = 3424  # 'sunset</w>'
        output[8] = PAD_TOKEN  # EOS token for both models
        return output.reshape([1, 77])

    def unconditional_tokens(self):
        MAX_LENGTH = 77  # Depend on model setting
        BOS_TOKEN = 49406  # Depend on model setting
        PAD_TOKEN = 49407  # Depend on model setting
        output = np.full(MAX_LENGTH, PAD_TOKEN, dtype=np.int32)
        output[0] = BOS_TOKEN
        output[1] = PAD_TOKEN  # EOS token for both models
        return output.reshape([1, 77])

    def embedding(self, input_ids):
        # Prepare input tensors
        print(input_ids)
        output_tensors = self.sd_executor.run(
            ["last_hidden_state", "pooler_output"], {
                'input_ids': input_ids,
            }
        )
        return output_tensors[0]


class VAE:
    def __init__(self, sd_vae_config, sd_executor):
        self.sd_vae_config = sd_vae_config
        self.sd_executor = sd_executor

    def decode(self, latents):
        if not latents.size > 0:
            return np.array([], dtype=dtype)

        input_tensors = latents * (1.0 / self.sd_vae_config['sd_decode_scale_strength'])
        print("latent_sample shape:", input_tensors.shape)
        output_tensors = self.sd_executor.run(
            ['sample'], {
                'latent_sample': input_tensors.astype(np.float32),
            }
        )
        result = output_tensors[0] / 2.0 + 0.5
        return result


class ImageData:
    def __init__(self, data, size):
        self.data = data
        self.size = size


def convert_result(tensor):
    shape = tensor.shape

    if len(shape) != 4:
        raise ValueError("Expected 4D tensor (N, C, H, W)")

    batch_size, channels, height, width = shape

    if batch_size != 1:
        raise ValueError("Batch size > 1 is not supported")

    image_size = height * width * channels
    tensor_data = tensor.flatten()
    image_data = np.zeros(image_size, dtype=np.uint8)

    for c in range(channels):
        for h in range(height):
            for w in range(width):
                tensor_at = (c * height + h) * width + w
                cur_pixel = (h * width + w) * channels + c
                image_data[cur_pixel] = np.round(
                    np.clip(tensor_data[tensor_at], 0.0, 1.0) * 255
                ).astype(np.uint8)

    return ImageData(image_data, image_size)


class CommandLineInput:
    def __init__(self, output_path, sd_input_width, sd_input_height, sd_input_channel):
        self.output_path = output_path
        self.sd_input_width = sd_input_width
        self.sd_input_height = sd_input_height
        self.sd_input_channel = sd_input_channel


def save_image(params, image_data):
    if image_data is None:
        print("generate failed")
        return

    last = params.output_path.rfind('.')
    file_name = params.output_path[:last] if last != -1 else params.output_path
    final_image_path = file_name + ".png"

    # Reshape the image data to the correct shape
    image_data = image_data.reshape((params.sd_input_height, params.sd_input_width, params.sd_input_channel))

    # Convert the image data to a PIL Image and save it
    image = Image.fromarray(image_data, 'RGB')
    image.save(final_image_path)

    print("\n")
    print(f"save result image to '{final_image_path}'")
    print("\n")


# ============================================== begin test ==============================================
# test config
onnx_unet_model = ort.InferenceSession(
    "/Volumes/AL-Data-W04/WorkingSpace/Self-Project/project-onnx-sd/sd/sd-base-model/onnx-official-sd-v15/unet/model.onnx")
onnx_clip_model = ort.InferenceSession(
    "/Volumes/AL-Data-W04/WorkingSpace/Self-Project/project-onnx-sd/sd/sd-base-model/onnx-official-sd-v15/text_encoder/model.onnx")
onnx_vae_encoder_model = ort.InferenceSession(
    "/Volumes/AL-Data-W04/WorkingSpace/Self-Project/project-onnx-sd/sd/sd-base-model/onnx-official-sd-v15/vae_encoder/model.onnx")
onnx_vae_decoder_model = ort.InferenceSession(
    "/Volumes/AL-Data-W04/WorkingSpace/Self-Project/project-onnx-sd/sd/sd-base-model/onnx-official-sd-v15/vae_decoder/model.onnx")

sd_vae_config = {
    'sd_decode_scale_strength': 0.18215
}

sd_unet_config = {
    'sd_input_width': 64,
    'sd_input_height': 64,
    'sd_input_channel': 4,
    'sd_scale_guidance': 1.0,
    'sd_inference_steps': 3
}

scheduler_config = {
    'scheduler_training_steps': 1000,
    'scheduler_beta_start': 0.00085,
    'scheduler_beta_end': 0.012,
    'scheduler_beta_type': 'BETA_TYPE_SCALED_LINEAR',
    # in 'BETA_TYPE_LINEAR', 'BETA_TYPE_SCALED_LINEAR', 'BETA_TYPE_SQUAREDCOS_CAP_V2'
    'scheduler_alpha_type': 'ALPHA_TYPE_COSINE',  # in 'ALPHA_TYPE_COSINE', 'ALPHA_TYPE_EXP'
    'scheduler_predict_type': 'PREDICT_TYPE_EPSILON',
    # in 'PREDICT_TYPE_EPSILON', 'PREDICT_TYPE_V_PREDICTION', 'PREDICT_TYPE_SAMPLE'
    'scheduler_seed': 15
}

# test input
encoded_img = np.array([])

# test inference
sd_scheduler = SchedulerBase(scheduler_config)
unet = UNet(sd_unet_config, sd_scheduler, onnx_unet_model)
clip = Clip(onnx_clip_model)
vae_decoder = VAE(sd_vae_config, onnx_vae_decoder_model)
embs_positive = clip.embedding(clip.conditional_tokens())  # default -p "A cat in the water at sunset" in idx
embs_negative = np.array([])  # default currently no uncondational input
latent_result = unet.inference(embs_positive, embs_negative, encoded_img)
decode_result = vae_decoder.decode(latent_result)
output_image = convert_result(decode_result)

# test output
params = CommandLineInput(
    "/Volumes/AL-Data-W04/WorkingSpace/Self-Project/project-onnx-sd/sd/io-test/output.png",
    512, 512, 3)
save_image(params, output_image.data)
# ============================================== after test ==============================================
