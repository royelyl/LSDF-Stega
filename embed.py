import os
import sys
import torch
import transformers
import accelerate
import numpy as np
from termcolor import colored
import time
import json
import random
import math
import logging
from tqdm.auto import tqdm
from argparse import Namespace
from huggingface_hub.file_download import hf_hub_download
from accelerate import Accelerator
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)
from nltk import sent_tokenize
import nltk

nltk.download('punkt')

"""
超参数：
batch_size -- 批次大小，默认为1，不需要修改
unit_seq_len -- 生成句子长度，默认为25
embedding_size -- 嵌入维度数
"""
batch_size = 1
unit_seq_len = 25
embedding_size = 50265

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

    
## 获得噪声时间表
def get_time_variables(t, total_t, device):  # cosine schedule

    def ft(small_t, big_t, s=1e-4):
        return torch.cos((small_t / big_t + s) / (1 + s) * math.pi / 2) ** 2

    alpha_t_bar = ft(t, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    alpha_t_minus_bar = ft(t - 1, total_t) / ft(torch.zeros(t.shape).to(device), total_t)
    beta_t = 1 - (alpha_t_bar / alpha_t_minus_bar)
    beta_t_til = (1 - alpha_t_minus_bar) / (1 - alpha_t_bar) * beta_t
    alpha_t = 1 - beta_t
    return alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t

## 添加控制转移
def apply_controlling_drift(args, perturbed_inputs_diralpha):
    if args.decode_ctr_lr <= 0:
        args.ctr_loss = -1
        return perturbed_inputs_diralpha

    if args.ctr_model is None:
        args.ctr_model = AutoModelForSequenceClassification.from_pretrained(args.ctr_model_name).to(args.accelerator.device)
    optimizing_label_index = args.ctr_opt_label_idx

    for ctr_i in range(1):
        with torch.enable_grad():
            perturbed_inputs_diralpha_4ctr = perturbed_inputs_diralpha.clone()
            perturbed_inputs_diralpha_4ctr.requires_grad_()
            perturbed_inputs_simplex_4ctr = torch.nn.functional.softmax(perturbed_inputs_diralpha_4ctr, dim=-1)
            perturbed_inputs_embeds_4ctr = torch.nn.functional.linear(perturbed_inputs_simplex_4ctr, args.ctr_model.get_input_embeddings().weight.t())
            ctr_loss = -torch.nn.functional.log_softmax(args.ctr_model(inputs_embeds=perturbed_inputs_embeds_4ctr).logits, dim=-1)[:,optimizing_label_index].mean()
            args.ctr_loss = ctr_loss
            ctr_delta = -torch.autograd.grad(ctr_loss, perturbed_inputs_diralpha_4ctr)[0]
        perturbed_inputs_diralpha = perturbed_inputs_diralpha + args.decode_ctr_lr * ctr_delta # we use a fixed balancing factor in this work, which can be improved in the future

    return perturbed_inputs_diralpha

# 采样映射
def logits_sampling_projection(logits, top_p, one_hot_value):

    assert len(logits.size()) == 3
    very_low_value = -10000

    probs = torch.nn.functional.softmax(logits, dim=-1)
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cum_sum_probs = torch.cumsum(sorted_probs, dim=-1)
    nucleus = cum_sum_probs < top_p
    nucleus = torch.cat([nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1)
    valid_indices = nucleus.scatter(2, indices, nucleus)

    filtered_logits = logits.masked_fill(valid_indices == 0, -float('Inf'))
    m = torch.distributions.categorical.Categorical(logits=filtered_logits)
    selected = m.sample()
    return 2 * one_hot_value * torch.nn.functional.one_hot(selected, logits.size(2)) - one_hot_value


def decode(args, batch_input_ids, dec_depth, total_t, model_embedding_lut, embedding_sum_layer, timestep_layer, model,
           tokenizer,message_path,output_dir):
    if args.decode_truncate_len > 0:
        diffusion_input_ids = batch_input_ids[:, args.context_size:-args.decode_truncate_len]
    else:
        diffusion_input_ids = batch_input_ids[:, args.context_size:]

    assert (
                   args.max_seq_length - args.context_size - args.decode_truncate_len) % dec_depth == 0, "check whether the total generation length is divisible by the depth of decoding"
    unit_seq_len = int((args.max_seq_length - args.context_size - args.decode_truncate_len) / dec_depth)
    if args.context_size > 0:
        unit_context_input_ids = batch_input_ids[:, :args.context_size].clone()
    else:
        unit_context_input_ids = None
    history_decode_ids = None

    for i in range(dec_depth):
        # 在这里进行隐写
        # 这里选择嵌入的维度大小

        img, binary_random = stega_generator(batch_size, unit_seq_len, embedding_size, message_path, output_dir)
        stega_noise = img.to(dtype=torch.float32)


        # 生成文本
        unit_noise = args.one_hot_value * stega_noise.to(args.accelerator.device)
        # unit_noise = args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size)).to(
        #     args.accelerator.device)

        xt = unit_noise

        if unit_context_input_ids is not None:
            context_inputs_embeds = model_embedding_lut(unit_context_input_ids)
            # print("unit_context_input_ids is not None!")
            # print("context_inputs_embeds:", context_inputs_embeds)
        else:
            context_inputs_embeds = None
            # print("unit_context_input_ids is None!")

        t_range = list(range(1, total_t + 1))
        t_range.reverse()
        progress_bar = tqdm(range(len(t_range)), disable=not args.accelerator.is_local_main_process)

        ## 在这里进行初始化
        simplex_list = []

        for t in t_range:
            selected_t = torch.FloatTensor([t]).repeat(batch_size).to(args.accelerator.device)
            alpha_t_bar, alpha_t_minus_bar, beta_t, beta_t_til, alpha_t = get_time_variables(selected_t, total_t,
                                                                                             args.accelerator.device)
            zt = args.one_hot_value * torch.normal(0, 1, size=(batch_size, unit_seq_len, args.vocab_size)).to(
                args.accelerator.device)

            # 在t = 2时间步进行隐写
            if t == 2:
                zt = args.one_hot_value * stega_noise.to(args.accelerator.device) 

            #    print("z2:",zt)
                #torch.save(zt, f"./data_0924/z{t}.pt") 
                torch.save(zt, f"{output_dir}/z{t}.pt")

            perturbed_inputs_diralpha = xt
            perturbed_inputs_simplex = torch.nn.functional.softmax(perturbed_inputs_diralpha, dim=-1)

            perturbed_inputs_embeds = embedding_sum_layer(perturbed_inputs_simplex)
            t_progress = selected_t / total_t
            timestep_embeds = timestep_layer(t_progress.view(batch_size, 1, 1).repeat(1, unit_seq_len, 1))

            diffusion_embeds = perturbed_inputs_embeds + timestep_embeds
            if context_inputs_embeds is not None:
                diffusion_embeds = torch.cat((context_inputs_embeds, diffusion_embeds), dim=1)
            outputs = model(inputs_embeds=diffusion_embeds, output_hidden_states=False)
            equivalent_score = outputs.logits
            if unit_context_input_ids is not None:
                equivalent_score = equivalent_score[:, unit_context_input_ids.size(1):].contiguous()

            # controlled generation if the balancing factor > 0
            equivalent_score = apply_controlling_drift(args, equivalent_score)

            projected_logits = logits_sampling_projection(equivalent_score, top_p=args.projection_top_p,
                                                          one_hot_value=args.one_hot_value)

            xt = torch.sqrt(alpha_t_minus_bar).view(-1, 1, 1) * projected_logits
            xt = xt + torch.sqrt(1 - alpha_t_minus_bar).view(-1, 1, 1) * zt

            # 保存隐写的xt
            if t == 2:
                #torch.save(xt, f"./data_0924/stego_x{t}.pt")
                torch.save(xt, f"{output_dir}/stego_x{t}.pt")
            #
            # 保存模型超参数
            if t == 2:
                #torch.save(alpha_t_minus_bar, "./data_0924/alpha_t_minus_bar.pt")
                #torch.save(projected_logits, "./data_0924/projected_logits_2.pt")
                torch.save(alpha_t_minus_bar, f"{output_dir}/alpha_t_minus_bar.pt")
                torch.save(projected_logits, f"{output_dir}/projected_logits_2.pt")

            progress_bar.update(1)

            if t % args.decode_log_interval == 0 or t == 1:
                simplex = torch.nn.functional.softmax(xt, dim=-1)

                simplex_list = simplex.detach().cpu().numpy()

                # ## 在这里保存一些必要信息并写入txt文件
                # # 打开文件以追加模式添加新行
                # with open("output.txt", "a") as file:
                #     for line in simplex_list:
                #         file.write(line)

                logger.info(f"noise coef at t: {torch.sqrt(1 - alpha_t_bar).item()}")

                if unit_context_input_ids is not None:
                    context_sequences = tokenizer.batch_decode(unit_context_input_ids.detach().to('cpu'))
                    logger.info(f"context: {context_sequences}")

                real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach().to('cpu'))
                logger.info(f"t={t} (argmax w_t-1): {colored(str(sampled_sequences), 'red')}")

                simplex = equivalent_score
                real_token_ids_list = torch.argmax(simplex, dim=-1).view(batch_size, unit_seq_len)
                sampled_sequences = tokenizer.batch_decode(real_token_ids_list.clone().detach().to('cpu'))
                logger.info(f"t={t} (argmax w_logits): {colored(str(sampled_sequences), 'blue')}")

                alt_i = 1  # look at the second best candidate; note that the whole sequence is not meaningful; each token can be considered as a substitution for the corresponding token in the argmax sequence
                alt_real_token_ids_list = torch.topk(simplex, alt_i + 1, dim=-1).indices[:, :, alt_i].view(batch_size,
                                                                                                           unit_seq_len)
                alt_sampled_sequences = tokenizer.batch_decode(alt_real_token_ids_list.clone().detach().to('cpu'))
                logger.info(f"t={t} (argsecondmax w_logits): {alt_sampled_sequences}")

                logger.info(f"ctr loss: {args.ctr_loss}")

        unit_context_input_ids = torch.cat((unit_context_input_ids, real_token_ids_list), dim=1)
        if history_decode_ids is None:
            history_decode_ids = real_token_ids_list
        else:
            history_decode_ids = torch.cat((history_decode_ids, real_token_ids_list), dim=1)

    if args.context_size > 0:
        init_context_input_ids = batch_input_ids[:, :args.context_size].clone()
        context_sequences = tokenizer.batch_decode(init_context_input_ids.detach().to('cpu'))
    else:
        init_context_input_ids = None
        context_sequences = None
    gold_sequences = tokenizer.batch_decode(diffusion_input_ids.clone().detach().to('cpu'))
    sampled_sequences = tokenizer.batch_decode(history_decode_ids.clone().detach().to('cpu'))
    logger.info(f"context: {context_sequences}")
    logger.info(f"gold: {colored(str(gold_sequences), 'yellow')}")
    logger.info(f"generation: {colored(str(sampled_sequences), 'red')}")

    return history_decode_ids, init_context_input_ids, diffusion_input_ids, sampled_sequences, context_sequences, gold_sequences


def read_dat_file(file_path):
    with open(file_path, 'rb') as file:  # 以二进制模式读取文件
        content = file.read()  # 读取文件内容
    return content

def to_binary_string(data):
    # 将字节数据转换为二进制字符串
    return ''.join(format(byte, '08b') for byte in data)

def stega_generator(batch_size, unit_seq_len, vocab_size, message_path, output_dir):
    """
    文本隐写函数

    参数：
    embedding_rate -- 嵌入率
    """

    embedding_rate = 1

    print("开始进行文本隐写！")
    total_elements = batch_size * unit_seq_len * vocab_size
    shape = (batch_size, unit_seq_len, vocab_size)
    print(f"批次大小为：{batch_size},句子长度为：{unit_seq_len}，词汇表维度为：{vocab_size}, 二进制比特数{total_elements}")
    #array = np.array([])
    array = []
    def generate_binary_random(bits):
        # 生成一个 bits 位的随机整数

        random_integer = random.randint(0, 2 ** bits - 1)

        # 将整数转换为二进制字符串，并去掉前缀'0b'
        binary_string = bin(random_integer)[2:]

        # 如果生成的二进制位数不足 bits 位，前面补零
        binary_string = binary_string.zfill(bits)

        return binary_string

    def xor_binary_strings(binary_str1, binary_str2):
        # 确保两个二进制字符串长度相同
        length = max(len(binary_str1), len(binary_str2))
        binary_str1 = binary_str1.zfill(length)
        binary_str2 = binary_str2.zfill(length)

        # 将二进制字符串转换为整数，进行异或操作，然后再转换为二进制字符串
        result_integer = int(binary_str1, 2) ^ int(binary_str2, 2)
        result_binary = bin(result_integer)[2:].zfill(length)

        return result_binary

    def normal_distribution(x, mean, std_dev):
        # 正态分布的概率密度函数
        return (1 / (std_dev * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std_dev) ** 2)

    def rejection_sampling_normal(mean, std_dev, a, b, decimal_places):
        sample = None

        while sample is None or not (a <= sample < b):
            x_candidate = round(random.normalvariate(mean, std_dev), decimal_places)
            acceptance_prob = normal_distribution(x_candidate, mean, std_dev) / normal_distribution(mean, mean, std_dev)

            # 接受样本的条件
            if random.uniform(0, 1) < acceptance_prob:
                sample = x_candidate

        return sample



    # 生成一个n * embedding_rate位的二进制随机数
    # batch_size * sentence_length * embedding_size
    dat_data = read_dat_file(message_path)
    dat_binary = to_binary_string(dat_data)
    #print("这里————————————————",message_path,dat_binary)
    num_repeats = math.ceil(total_elements * embedding_rate / len(dat_binary))
    # 生成新的二进制字符串
    binary_random = dat_binary * num_repeats
    #print("截取之前的长度",len(dat_binary))
    binary_random = binary_random[:(total_elements * embedding_rate)]
    #print("更新后的 dat_binary:", num_repeats, len(dat_binary))
    #binary_random = generate_binary_random(total_elements * embedding_rate)
    print("二进制随机比特流生成完毕！")
    print("嵌入率为1bit！")
    print("随机生成的二进制序列为：",binary_random[0:32])
    task = os.path.basename(message_path).split('.')[0]
    with open(f"{output_dir}/{task}.bit", "w") as file:
        file.write(binary_random)

    # 定义秘密信息序列

    # result = xor_binary_strings(binary_random, binary_sequence)
    result = binary_random
    print("秘密信息与随机比特流异或完毕！")
    # print("异或得到的二进制序列为：",result)


    # 每p位消息为1组，组成n/p组

    group_result = [result[i: i + embedding_rate] for i in range(0, len(result), embedding_rate)]
    # G中的每一组i，先利用自适应算术解码求出对应的区间，然后在对应的区间中随机采样z,直到z落在该区间中
    for i, group in enumerate(group_result, start=1):
        # print(f"Group {i}: {group}")
        # 将二进制串转化为小数形式，"1100"转换为"0.1100"
        m = 0
        for j in range(1,len(group) + 1):
            m += int(group[j - 1]) * (2 ** (-1 * j))
        # print("Group Result:", m)
        # 自适应算术解码
        encoder = AdaptiveArithmeticCoding()
        for symbol in group:
            encoder.encode_symbol(symbol)
        # print("Encoded value:", encoder.low, encoder.high)
        # 拒绝采样
        mean, std_dev = 0, 1  # 正态分布的均值和标准差
        start, end = encoder.low, encoder.high  # 指定区间
        decimal_places = 4  # 指定小数位数
        me = rejection_sampling_normal(mean, std_dev, start, end, decimal_places)
        sign = random.choice([-1, 1])
        me = me * sign
        # print("z:",me)
        #array = np.append(array,me)
        array.append(me)
    # print(array)
    print("隐变量生成完毕！")
    array = np.array(array)
    torch_tensor = torch.from_numpy(array).reshape(*shape)
    #print(torch_tensor.shape)
    #print(torch_tenso
    return torch_tensor, binary_random


class AdaptiveArithmeticCoding:
    def __init__(self):
        self.low = 0.0
        self.high = 1.0
        self.range = 1.0
        self.total_bits = 0
        self.probabilities = {'0': 0.5, '1': 0.5}
        self.nums = {'0': 1, '1': 1}
        self.bits = []

    def update_probabilities(self, symbol):
        self.nums[symbol] += 1
        total_count = sum(self.nums.values())
        for key in self.probabilities:
            self.probabilities[key] = self.nums[key] / total_count
            #print(self.probabilities[key])


    def encode_symbol(self, symbol):
        low_range = self.low + self.range * sum(self.probabilities[key] for key in self.probabilities if key < symbol)
        high_range = low_range + self.range * self.probabilities[symbol]

        self.low = low_range
        self.high = high_range
        self.range = high_range - low_range
        self.total_bits += 1

        self.update_probabilities(symbol)


    def decode_symbol(self, value, bit_num):
        symbol = None
        for i in range(bit_num):
            for key in self.probabilities:
                low_range = self.low + self.range * sum(self.probabilities[k] for k in self.probabilities if k < key)
                high_range = low_range + self.range * self.probabilities[key]
                # print(low_range, high_range)
                if low_range <= value <= high_range:
                    symbol = key
                    self.bits.append(symbol)
                    break

            self.low = low_range
            self.high = high_range
            self.range = high_range - low_range
            self.total_bits += 1

            self.update_probabilities(symbol)

        return self.bits

def check_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
        print('A folder called "{}" is created.'.format(dir))

def test_text_embed(data):
    task_id = data["task_id"]
    message_list = data["message_paths"] 
    gen_num = len(message_list)
    task = ""
    output_dir = data["output"]  +'/'+ task_id
    print("output_dir",output_dir)
    # 底层模型参数，无需改动
    args = Namespace()
    args.model_name_or_path = "./pytorch"
    print(args.model_name_or_path)
    args.max_seq_length = 200
    args.one_hot_value = 5
    args.decoding_block_size = 5
    args.decode_total_gen_len = 10  # should be divisible by decode_depth
    args.decode_depth = 1
    
    args.decode_log_interval = 100
    args.total_t = 1000
    args.projection_top_p = 0.95
    args.seed = 2022
    args.decode_ctr_lr = 0.0  # set to 0 for unconstrained generation, large value for controlled generation
    args.use_slow_tokenizer = True
    
    # 隐写参数
    embedding_rate = 1 # 嵌入率
    embedding_size = 50265 # 嵌入维度
    n_examples = 1 # 生成样本句子数
    #DataSet = 'news' # 数据集
    #file_path = "./Data/news2020.txt"
    DataSet = 'zyy' # 数据集
    file_path = "./Data/zyy.txt"

    # 配置多GPU环境Accelerator
    accelerator = Accelerator()
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # 加载config及模型参数
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path, from_tf=False, config=config)
    # model = AutoModelForMaskedLM.from_pretrained("./train_news/save5000", from_tf=False, config=config)

    model.resize_token_embeddings(len(tokenizer))
    vocab_size = model.get_input_embeddings().weight.size(0)
    hidden_size = model.get_input_embeddings().weight.size(1)

    embedding_sum_layer = torch.nn.Linear(vocab_size, hidden_size, bias=False)
    _stdict = torch.load(os.path.join(args.model_name_or_path, "embed_sum_layer.pt"))
    _stdict = dict(
        (_k[len("module."):], _stdict[_k]) if _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
    embedding_sum_layer.load_state_dict(_stdict)

    timestep_layer = torch.nn.Linear(1, hidden_size, bias=True)
    _stdict = torch.load(os.path.join(args.model_name_or_path, "timestep_layer.pt"))
    _stdict = dict(
        (_k[len("module."):], _stdict[_k]) if _k.startswith("module.") else (_k, _stdict[_k]) for _k in _stdict)
    timestep_layer.load_state_dict(_stdict)

    # 准备模型
    model, embedding_sum_layer, timestep_layer = accelerator.prepare(model, embedding_sum_layer, timestep_layer)

    # a bit more preparation before decoding
    model.eval()
    model_embedding_lut = accelerator.unwrap_model(model).get_input_embeddings()
    args.vocab_size = vocab_size
    args.accelerator = accelerator
    args.ctr_model = None
    args.orig_decode_truncate_len = args.max_seq_length - args.decode_total_gen_len

    # start sampling from SSD-LM

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # sequences_list = []
    # history_decode_ids, context_input_ids, diffusion_input_ids, sampled_sequences, context_sequences, gold_sequences = \
    #          decode(args, input_ids, args.decode_depth, args.total_t, model_embedding_lut, embedding_sum_layer, timestep_layer, model, tokenizer)
    # sequences_list.append(context_sequences + sampled_sequences)
    # print(sequences_list)

    # for i in range(1000):
    #     history_decode_ids, context_input_ids, diffusion_input_ids, sampled_sequences, context_sequences, gold_sequences = \
    #         decode(args, input_ids, args.decode_depth, args.total_t, model_embedding_lut, embedding_sum_layer, timestep_layer, model, tokenizer)
    #     sequences_list.append(context_sequences + sampled_sequences)
    #     print(sampled_sequences)
    #     with open(os.path.join("./Result/News/4bit", "sampled_sequences_stega.txt"), 'a') as f:
    #         f.write(f"{sampled_sequences}\n")
    # print(sequences_list)
    # with open('stega.txt', 'w') as file:
    #     for item in sequences_list:
    #         file.write(f"{item}\n")

    def read_text_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:  # 确保使用正确的编码
            lines = file.readlines()  # 读取所有行，每行作为一个元素存储在列表中
        return lines
    
    dataset = read_text_file(file_path)
    
    th = 0
    while th < gen_num:
        save_data_main_dir = os.path.join(
            'data_0924', 'text',
            '{}_{}_{}bit'.format(DataSet.split('/')[-1],
                                 str('gpu').replace(':', '-'), embedding_size * embedding_rate))
        check_dir(save_data_main_dir)
        task = os.path.basename(message_list[th]).split('.')[0]
        print("---------now---------",task)
        save_context_path = os.path.join("./data_0924", 'news_{:.1f}bit_context.txt'.format(embedding_rate * embedding_size))

        #save_stego_path = os.path.join("./data_0924", 'news_{:.1f}bit_stego.txt'.format(embedding_rate * embedding_size))

        save_stego_path = output_dir+ f'/{task}/{task}.txt'
        dir_stego = output_dir+ f'/{task}'
        if not os.path.exists(dir_stego):              
            os.makedirs(dir_stego)
        print("save_stego_path",save_stego_path)
        f_context = open(save_context_path, 'w')
        f_stego = open(save_stego_path, 'w')
        context_lst = []
        stego_lst = []

        for i in tqdm(range(n_examples), ncols=70):
            # random.seed(os.urandom(1))
            #message_start_index = random.randint(0, 10000)
            message_start_index = 100

            context = dataset[i + n_examples]
            context = context.replace('<br /><br />', ' ').replace('<br />', ' ')  # remove all '<br />'
            context = ' '.join(sent_tokenize(context)[:3])  # Selecting leading 3 sentences as `context`
            # example = encode_text(model, tokenizer, message[message_start_index:], context, settings)
            prompt = context
            input_ids = torch.LongTensor(tokenizer.encode(prompt, add_special_tokens=False)).to(args.accelerator.device)
            args.context_size = len(input_ids)
            assert args.max_seq_length - args.decode_total_gen_len - args.context_size > 0, "check the length of the prompt"
            args.decode_truncate_len = args.orig_decode_truncate_len - args.context_size
            input_ids = input_ids.unsqueeze(0)
            history_decode_ids, context_input_ids, diffusion_input_ids, sampled_sequences, context_sequences, gold_sequences = \
                decode(args, input_ids, args.decode_depth, args.total_t, model_embedding_lut, embedding_sum_layer,
                       timestep_layer, model, tokenizer,message_list[0],dir_stego)
            context_lst.append(context_sequences)
            stego_lst.append(sampled_sequences)
            print(context_sequences)
            f_context.write(f"{context_sequences}\n")
            print(sampled_sequences)
            f_stego.write(f"{sampled_sequences}\n")
            th += 1

# for item in context_lst:
#     f_context.write(f"{item}\n")
# for item in stego_lst:
#     f_stego.write(f"{item}\n")
# f_context.write('\n'.join(context_lst))
# f_stego.write('\n'.join(stego_lst))


# # 所有的.dat文件都在autodl-tmp/LSDF/data_0924/目录下
# data_dir = "data_0924/"
# # 获取该目录下所有的.dat文件
# message_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.dat')]
# data = {"task_id": "3", "message_paths":message_paths, "output": "./output/embed"} 
# test_text_embed(data)