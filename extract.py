import torch
import random
import os
#TODO: 提取函数

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
            # print(self.probabilities[key])

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


def test_text_extract(data):
    task_id = data["task_id"]
    message_list = data["message_paths"]

    task = "" 
    gen_num = len(message_list)
    th = 0
    while th < gen_num:
        #task = os.path.basename(message_list[th]).split('.')[0]
        task = os.path.basename(message_list[th])
        output_dir = data["output"]  +'/extract_'+ task_id + f'/{task}'
        print("task",task)
        stego_x2 = torch.load(f"{message_list[th]}/stego_x2.pt",map_location=torch.device('cpu'))

        alpha_t_minus_bar = torch.load(f"{message_list[th]}/alpha_t_minus_bar.pt",map_location=torch.device('cpu'))

        projected_logits_2 = torch.load(f"{message_list[th]}/projected_logits_2.pt",map_location=torch.device('cpu'))

        zt = torch.load(f"{message_list[th]}/z2.pt",map_location=torch.device('cpu'))


        zs = (stego_x2 - torch.sqrt(alpha_t_minus_bar).view(-1, 1, 1)  * projected_logits_2) / (torch.sqrt(1 - alpha_t_minus_bar).view(-1, 1, 1))

        img = zs.to(dtype=torch.float32)

        # 还原为正值
        sign = random.choice([-1, 1])
        img = abs(img) / 5
        print(img)

        # 改变tensor的形状为1维
        torch_tensor = img.reshape(-1).numpy()
        print(torch_tensor.shape)

        merge_bits = []
        for temp in torch_tensor:
            decoder = AdaptiveArithmeticCoding()
            # print(temp)
            # print(decoder.decode_symbol(temp, 4))
            merge_bits += decoder.decode_symbol(temp, 1)
        print(len(merge_bits))
        print(merge_bits[0:32])
        result = ''.join(merge_bits)

        if not os.path.exists(output_dir):              
            os.makedirs(output_dir)

        with open(output_dir + f'/{task}.bit', 'w') as file:
            # 将变量写入文件
            file.write(result)
        th += 1
        
#data_json = {"task_id": "7", "message_paths": ["./output/embed/1/0001","./output/embed/1/0002"], "output": "./output/extract"}
#test_text_extract(data_json)