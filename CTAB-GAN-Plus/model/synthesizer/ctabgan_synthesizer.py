import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim 
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss,LayerNorm)
from model.synthesizer.transformer import ImageTransformer,DataTransformer
from model.synthesizer.transformer import ImageTransformer,DataTransformer 
from tqdm import tqdm
import optuna 

class Classifier(Module):
    def __init__(self, input_dim, dis_dims, st_ed,
                 dropout_rate=0.5, 
                 leaky_relu_slope_classifier=0.2): 
        super(Classifier, self).__init__()
        dim = input_dim - (st_ed[1] - st_ed[0])
        seq = []
        self.str_end = st_ed 

        for item in list(dis_dims):
            seq += [
                Linear(dim, item),
                LeakyReLU(leaky_relu_slope_classifier, inplace=True), 
                Dropout(dropout_rate) 
            ]
            dim = item 

        target_dim_size = st_ed[1] - st_ed[0]
        if target_dim_size == 1: 
            seq += [Linear(dim, 1)]
        elif target_dim_size == 2: 
            seq += [Linear(dim, 1), Sigmoid()]
        else: 
            seq += [Linear(dim, target_dim_size)]

        self.seq = Sequential(*seq) 

    def forward(self, input_data):
        label = None
        target_dim_size = self.str_end[1] - self.str_end[0]

        if target_dim_size == 1: 
            label = input_data[:, self.str_end[0]:self.str_end[1]]
        else: 
            label = torch.argmax(input_data[:, self.str_end[0]:self.str_end[1]], axis=-1)

        new_imp = torch.cat((input_data[:, :self.str_end[0]], input_data[:, self.str_end[1]:]), 1)

        output = self.seq(new_imp)
        if target_dim_size == 2 or target_dim_size == 1:
            return output.view(-1), label
        else: 
            return output, label

def apply_activate(data, output_info, gumbel_tau=0.2): 
    data_t = []
    st = 0 
    for item in output_info:
        ed = st + item[0] 
        if item[1] == 'tanh': 
            data_t.append(torch.tanh(data[:, st:ed]))
        elif item[1] == 'softmax': 
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=gumbel_tau))
        st = ed
    return torch.cat(data_t, dim=1) 

def get_st_ed(target_col_index, output_info):
    st = 0 
    c = 0
    tc = 0 

    for item in output_info:
        if c == target_col_index: 
            break

        if item[1] == 'tanh':
            st += item[0]
            if item[2] == 'yes_g':
                c += 1
        elif item[1] == 'softmax':
            st += item[0]
            c += 1 
        tc += 1

    ed = st + output_info[tc][0] 

    return (st, ed)

def random_choice_prob_index_sampling(probs, col_idx):
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(pp)), p=pp))
    return np.array(option_list).reshape(col_idx.shape)

def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

def maximum_interval(output_info):
    max_interval = 0
    for item in output_info:
        max_interval = max(max_interval, item[0])
    return max_interval

class Cond(object):
    def __init__(self, data, output_info):
        self.model = [] 
        st = 0
        counter = 0 
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed

        self.interval = [] 
        self.n_col = 0 
        self.n_opt = 0 
        self.p = np.zeros((counter, maximum_interval(output_info)))
        self.p_sampling = []

        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0) 

                log_freq = np.log(tmp + 1)
                log_freq_normalized = log_freq / np.sum(log_freq)

                sampling_freq_normalized = tmp / np.sum(tmp)
                self.p_sampling.append(sampling_freq_normalized)

                self.p[self.n_col, :item[0]] = log_freq_normalized
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed

        self.interval = np.asarray(self.interval)

    def sample_train(self, batch):
        if self.n_col == 0:
            return None 

        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32') 
        mask = np.zeros((batch, self.n_col), dtype='float32') 
        mask[np.arange(batch), idx] = 1

        opt1prime = random_choice_prob_index(self.p[idx])

        for i in np.arange(batch):
            chosen_col_interval_start = self.interval[idx[i], 0]
            chosen_option_in_col = opt1prime[i]
            vec[i, chosen_col_interval_start + chosen_option_in_col] = 1

        return vec, mask, idx, opt1prime

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim 
from torch.optim import Adam
from torch.nn import functional as F
from torch.nn import (Dropout, LeakyReLU, Linear, Module, ReLU, Sequential,
Conv2d, ConvTranspose2d, Sigmoid, init, BCELoss, CrossEntropyLoss,SmoothL1Loss,LayerNorm)
from model.synthesizer.transformer import ImageTransformer,DataTransformer
from model.synthesizer.transformer import ImageTransformer,DataTransformer 
from tqdm import tqdm
import optuna 

class Classifier(Module):
    def __init__(self, input_dim, dis_dims, st_ed,
                 dropout_rate=0.5, 
                 leaky_relu_slope_classifier=0.2): 
        super(Classifier, self).__init__()
        dim = input_dim - (st_ed[1] - st_ed[0])
        seq = []
        self.str_end = st_ed 

        for item in list(dis_dims):
            seq += [
                Linear(dim, item),
                LeakyReLU(leaky_relu_slope_classifier, inplace=True), 
                Dropout(dropout_rate) 
            ]
            dim = item 

        target_dim_size = st_ed[1] - st_ed[0]
        if target_dim_size == 1: 
            seq += [Linear(dim, 1)]
        elif target_dim_size == 2: 
            seq += [Linear(dim, 1), Sigmoid()]
        else: 
            seq += [Linear(dim, target_dim_size)]

        self.seq = Sequential(*seq) 

    def forward(self, input_data):
        label = None
        target_dim_size = self.str_end[1] - self.str_end[0]

        if target_dim_size == 1: 
            label = input_data[:, self.str_end[0]:self.str_end[1]]
        else: 
            label = torch.argmax(input_data[:, self.str_end[0]:self.str_end[1]], axis=-1)

        new_imp = torch.cat((input_data[:, :self.str_end[0]], input_data[:, self.str_end[1]:]), 1)

        output = self.seq(new_imp)
        if target_dim_size == 2 or target_dim_size == 1:
            return output.view(-1), label
        else: 
            return output, label

def apply_activate(data, output_info, gumbel_tau=0.2): 
    data_t = []
    st = 0 
    for item in output_info:
        ed = st + item[0] 
        if item[1] == 'tanh': 
            data_t.append(torch.tanh(data[:, st:ed]))
        elif item[1] == 'softmax': 
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=gumbel_tau))
        st = ed
    return torch.cat(data_t, dim=1) 

def get_st_ed(target_col_index, output_info):
    st = 0 
    c = 0
    tc = 0 

    for item in output_info:
        if c == target_col_index: 
            break

        if item[1] == 'tanh':
            st += item[0]
            if item[2] == 'yes_g':
                c += 1
        elif item[1] == 'softmax':
            st += item[0]
            c += 1 
        tc += 1

    ed = st + output_info[tc][0] 

    return (st, ed)

def random_choice_prob_index_sampling(probs, col_idx):
    option_list = []
    for i in col_idx:
        pp = probs[i]
        option_list.append(np.random.choice(np.arange(len(pp)), p=pp))
    return np.array(option_list).reshape(col_idx.shape)

def random_choice_prob_index(a, axis=1):
    r = np.expand_dims(np.random.rand(a.shape[1 - axis]), axis=axis)
    return (a.cumsum(axis=axis) > r).argmax(axis=axis)

def maximum_interval(output_info):
    max_interval = 0
    for item in output_info:
        max_interval = max(max_interval, item[0])
    return max_interval

class Cond(object):
    def __init__(self, data, output_info):
        self.model = [] 
        st = 0
        counter = 0 
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                counter += 1
                self.model.append(np.argmax(data[:, st:ed], axis=-1))
                st = ed

        self.interval = [] 
        self.n_col = 0 
        self.n_opt = 0 
        self.p = np.zeros((counter, maximum_interval(output_info)))
        self.p_sampling = []

        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                continue
            elif item[1] == 'softmax':
                ed = st + item[0]
                tmp = np.sum(data[:, st:ed], axis=0) 

                log_freq = np.log(tmp + 1)
                log_freq_normalized = log_freq / np.sum(log_freq)

                sampling_freq_normalized = tmp / np.sum(tmp)
                self.p_sampling.append(sampling_freq_normalized)

                self.p[self.n_col, :item[0]] = log_freq_normalized
                self.interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self.n_col += 1
                st = ed

        self.interval = np.asarray(self.interval)

    def sample_train(self, batch):
        if self.n_col == 0:
            return None 

        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32') 
        mask = np.zeros((batch, self.n_col), dtype='float32') 
        mask[np.arange(batch), idx] = 1

        opt1prime = random_choice_prob_index(self.p[idx])

        for i in np.arange(batch):
            chosen_col_interval_start = self.interval[idx[i], 0]
            chosen_option_in_col = opt1prime[i]
            vec[i, chosen_col_interval_start + chosen_option_in_col] = 1

        return vec, mask, idx, opt1prime

    def sample(self, batch):
        if self.n_col == 0:
            return np.zeros((batch, 0), dtype='float32') 

        idx = np.random.choice(np.arange(self.n_col), batch)

        vec = np.zeros((batch, self.n_opt), dtype='float32')
        opt1prime = random_choice_prob_index_sampling(self.p_sampling, idx) 

        for i in np.arange(batch):
            chosen_col_interval_start = self.interval[idx[i], 0]
            chosen_option_in_col = opt1prime[i]
            vec[i, chosen_col_interval_start + chosen_option_in_col] = 1

        return vec

def cond_loss(data, output_info, c, m):
    loss_components = []
    st_data = 0 
    st_condition = 0 

    for item_idx, item_info in enumerate(output_info):
        if item_info[1] == 'tanh': 
            st_data += item_info[0]
            continue
        elif item_info[1] == 'softmax': 
            data_col_end = st_data + item_info[0]
            condition_col_end = st_condition + item_info[0]

            tmp_loss = F.cross_entropy(
                data[:, st_data:data_col_end],
                torch.argmax(c[:, st_condition:condition_col_end], dim=1),
                reduction='none' 
            )
            loss_components.append(tmp_loss)

            st_data = data_col_end
            st_condition = condition_col_end

    if not loss_components: 
        return torch.tensor(0.0, device=data.device)

    loss_stacked = torch.stack(loss_components, dim=1) 

    masked_loss_sum = (loss_stacked * m).sum()
    return masked_loss_sum / data.size(0)


class Sampler(object):
    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self.data = data
        self.model = []
        self.n_samples = len(data)

        st = 0
        for item_info in output_info:
            if item_info[1] == 'tanh':
                st += item_info[0]
                continue
            elif item_info[1] == 'softmax':
                ed = st + item_info[0]
                options_indices = []
                for option_j_offset in range(item_info[0]): 
                    option_selected_indices = np.nonzero(data[:, st + option_j_offset])[0]
                    options_indices.append(option_selected_indices)
                self.model.append(options_indices)
                st = ed

    def sample(self, n_to_sample, col_indices_to_condition_on, opt_indices_in_cols):
        if col_indices_to_condition_on is None: 
            idx = np.random.choice(np.arange(self.n_samples), n_to_sample)
            return self.data[idx]

        sampled_data_indices = []
        for condition_col_idx, condition_opt_idx in zip(col_indices_to_condition_on, opt_indices_in_cols):
            eligible_sample_indices = self.model[condition_col_idx][condition_opt_idx]
            if len(eligible_sample_indices) == 0:
                chosen_idx = np.random.choice(np.arange(self.n_samples))
            else:
                chosen_idx = np.random.choice(eligible_sample_indices)
            sampled_data_indices.append(chosen_idx)

        return self.data[sampled_data_indices]


class Discriminator(Module):
    def __init__(self, side, layers):
        super(Discriminator, self).__init__()
        self.side = side
        self.info_layer_idx = len(layers) - 2
        self.seq = Sequential(*layers) 
        self.seq_info = Sequential(*layers[:self.info_layer_idx]) 

    def forward(self, input_data):
        full_output = self.seq(input_data)
        info_output = self.seq_info(input_data)
        return full_output, info_output


class Generator(Module):
    def __init__(self, side, layers):
        super(Generator, self).__init__()
        self.side = side
        self.seq = Sequential(*layers) 

    def forward(self, input_):
        return self.seq(input_)


def determine_layers_disc(side, num_channels,
                          leaky_relu_slope=0.2, 
                          max_conv_blocks=3): 
    if not (4 <= side <= 64):
        raise ValueError(f"지원하는 side 값은 4에서 64 사이여야 합니다. 입력된 값: {side}")

    layer_dims = [(1, side)] 

    current_out_channels = num_channels
    current_out_side = side // 2

    for _ in range(max_conv_blocks):
        if current_out_side < 4: 
            break
        layer_dims.append((current_out_channels, current_out_side))
        current_out_channels *= 2
        current_out_side //= 2

    layers_D = []
    for i in range(len(layer_dims) -1):
        input_spec = layer_dims[i]
        output_spec = layer_dims[i+1]

        layers_D += [
            Conv2d(input_spec[0], output_spec[0], kernel_size=4, stride=2, padding=1, bias=False),
            LayerNorm([output_spec[0], output_spec[1], output_spec[1]]), 
            LeakyReLU(leaky_relu_slope, inplace=True),
        ]

    final_conv_in_channels = layer_dims[-1][0]
    final_conv_kernel_size = layer_dims[-1][1] 
    layers_D += [Conv2d(final_conv_in_channels, 1, kernel_size=final_conv_kernel_size, stride=1, padding=0),
                  ReLU(True)] 

    return layers_D


def determine_layers_gen(side, random_dim, num_channels,
                          gen_max_conv_blocks=3,
                          gen_leaky_relu_slope=0.2): 

    if not (4 <= side <= 64):
        raise ValueError(f"지원하는 side 값은 4에서 64 사이여야 합니다. 입력된 값: {side}")

    layer_dims = []
    layer_dims.append((1, side)) 

    current_channels = num_channels
    current_side = side // 2
    for _ in range(gen_max_conv_blocks):
        if current_side < 4: break
        layer_dims.append((current_channels, current_side))
        current_channels *= 2
        current_side //= 2

    layer_dims.reverse() 

    initial_out_channels = layer_dims[0][0] 
    initial_spatial_size = layer_dims[0][1] 

    layers_G = [
        ConvTranspose2d(
            in_channels=random_dim, 
            out_channels=initial_out_channels,
            kernel_size=initial_spatial_size, 
            stride=1,
            padding=0,
            bias=False
        )
    ]

    for i in range(len(layer_dims) - 1):
        input_block_spec = layer_dims[i] 
        output_block_spec = layer_dims[i+1] 

        in_ch = input_block_spec[0]
        in_size = input_block_spec[1]
        out_ch = output_block_spec[0]

        layers_G += [
            LayerNorm([in_ch, in_size, in_size]), 
            LeakyReLU(gen_leaky_relu_slope, inplace=True),
            ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=4,
                stride=2,
                padding=1, 
                bias=True 
            )
        ]
    return layers_G


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos(torch.clamp((low_norm * high_norm).sum(1), -1.0, 1.0)).unsqueeze(1)
    so = torch.sin(omega)
    res = torch.where(so.abs() < 1e-7,
                      low * (1.0 - val) + high * val, 
                      (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high) 
    return res

def calc_gradient_penalty_slerp(netD, real_data, fake_data, transformer, device='cpu', lambda_=10): 
    batchsize = real_data.shape[0]
    alpha = torch.rand(batchsize, 1, device=device)

    interpolates_1d = slerp(alpha, real_data, fake_data) 
    interpolates_1d = interpolates_1d.to(device)
    interpolates_img = transformer.transform(interpolates_1d) 
    interpolates_img.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates_img)

    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates_img,
        grad_outputs=torch.ones_like(disc_interpolates, device=device), 
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients_norm = gradients.view(batchsize, -1).norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1) ** 2).mean() * lambda_

    return gradient_penalty

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02) 
        if m.bias is not None:
            init.constant_(m.bias.data, 0) 
    elif classname.find('BatchNorm') != -1 or classname.find('LayerNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02) 
        init.constant_(m.bias.data, 0) 

class CTABGANSynthesizer:
    def __init__(self,
                 class_dim=(256, 256, 256, 256),
                 random_dim=100,
                 num_channels=64,
                 classifier_dropout_rate=0.5, 
                 leaky_relu_slope_classifier=0.2,

                 batch_size=500, 
                 epochs=300, 

                 lr_g=2e-4, lr_d=2e-4, lr_c=2e-4,
                 beta1_g=0.5, beta2_g=0.9, beta1_d=0.5, beta2_d=0.9, beta1_c=0.5, beta2_c=0.9,
                 eps_common=1e-8, l2scale_common=1e-5, 

                 lambda_gp=10.0, 

                 gumbel_tau=0.2, 
                 leaky_relu_slope=0.2, 
                 gen_leaky_relu_slope=0.2, 
                 disc_max_conv_blocks=3, 
                 gen_max_conv_blocks=3, 

                 lambda_cond_loss=1.0, 
                 lambda_info_loss=1.0, 
                 lambda_aux_classifier_loss=1.0, 

                 ci_discriminator_steps=5, 
                 condition_column=None,
                 verbose=False):

        self.class_dim = class_dim
        self.random_dim = random_dim
        self.num_channels = num_channels
        self.classifier_dropout_rate = classifier_dropout_rate
        self.leaky_relu_slope_classifier = leaky_relu_slope_classifier

        self.batch_size = batch_size
        self.epochs = epochs

        self.lr_g, self.lr_d, self.lr_c = lr_g, lr_d, lr_c
        self.beta1_g, self.beta2_g = beta1_g, beta2_g
        self.beta1_d, self.beta2_d = beta1_d, beta2_d
        self.beta1_c, self.beta2_c = beta1_c, beta2_c

        self.eps_common = eps_common
        self.l2scale_common = l2scale_common

        self.lambda_gp = lambda_gp

        self.gumbel_tau = gumbel_tau
        self.leaky_relu_slope = leaky_relu_slope
        self.gen_leaky_relu_slope = gen_leaky_relu_slope
        self.disc_max_conv_blocks = disc_max_conv_blocks
        self.gen_max_conv_blocks = gen_max_conv_blocks

        self.lambda_cond_loss = lambda_cond_loss
        self.lambda_info_loss = lambda_info_loss
        self.lambda_aux_classifier_loss = lambda_aux_classifier_loss

        self.ci_discriminator_steps = ci_discriminator_steps

        self.verbose = verbose

        self.dside = None
        self.gside = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.verbose:
            print(f"\nPyTorch 장치 확인: {self.device}를 사용하여 학습을 진행합니다.")
            if not torch.cuda.is_available(): print("경고: CUDA 사용 불가. CPU로 학습합니다.")

        self.G_losses, self.D_losses, self.C_losses = [], [], []

    def fit(self, train_data_df=pd.DataFrame(), categorical_columns=[], mixed_columns={},
            general_columns=[], non_categorical_columns=[], type_dict={}, trial=None):
        
        problem_type, target_index = None, None
        if type_dict:
            problem_type = list(type_dict.keys())[0]
            if problem_type:
                try: target_index = train_data_df.columns.get_loc(type_dict[problem_type])
                except KeyError:
                    if self.verbose: print(f"경고: 타겟 컬럼 '{type_dict[problem_type]}' 찾을 수 없음.")
                    target_index = None

        self.transformer = DataTransformer(train_data=train_data_df.copy(),
                                           categorical_list=categorical_columns, mixed_dict=mixed_columns,
                                           general_list=general_columns, non_categorical_list=non_categorical_columns)
        self.transformer.fit()
        transformed_train_data = self.transformer.transform(train_data_df.values)

        data_sampler = Sampler(transformed_train_data, self.transformer.output_info)
        data_dim = self.transformer.output_dim
        self.cond_generator = Cond(transformed_train_data, self.transformer.output_info)

        sides = [4, 8, 16, 24, 32, 64] 
        col_size_d = data_dim + self.cond_generator.n_opt
        self.dside = next((s for s in sides if s * s >= col_size_d), sides[-1])
        col_size_g = data_dim
        self.gside = next((s for s in sides if s * s >= col_size_g), sides[-1])

        layers_G = determine_layers_gen(
            self.gside,
            self.random_dim + self.cond_generator.n_opt, 
            self.num_channels,
            self.gen_max_conv_blocks,
            gen_leaky_relu_slope=self.gen_leaky_relu_slope
        )
        layers_D = determine_layers_disc(
            self.dside,
            self.num_channels,
            self.leaky_relu_slope, 
            self.disc_max_conv_blocks
        )

        self.generator = Generator(self.gside, layers_G).to(self.device)
        discriminator = Discriminator(self.dside, layers_D).to(self.device)

        optimizerG = Adam(self.generator.parameters(), lr=self.lr_g, betas=(self.beta1_g, self.beta2_g), eps=self.eps_common, weight_decay=self.l2scale_common)
        optimizerD = Adam(discriminator.parameters(), lr=self.lr_d, betas=(self.beta1_d, self.beta2_d), eps=self.eps_common, weight_decay=self.l2scale_common)

        classifier, optimizerC, st_ed = None, None, None
        if target_index is not None and problem_type:
            st_ed = get_st_ed(target_index, self.transformer.output_info)
            classifier = Classifier(data_dim, self.class_dim, st_ed,
                                    self.classifier_dropout_rate,
                                    self.leaky_relu_slope_classifier
                                   ).to(self.device)
            optimizerC = Adam(classifier.parameters(), lr=self.lr_c, betas=(self.beta1_c, self.beta2_c), eps=self.eps_common, weight_decay=self.l2scale_common)

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)
        if classifier: classifier.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside) 
        self.Dtransformer = ImageTransformer(self.dside) 

        steps_per_epoch = max(1, len(transformed_train_data) // self.batch_size)

        for epoch_idx in tqdm(range(self.epochs), desc="Training Epochs"):
            epoch_g_loss_sum, epoch_d_loss_sum, epoch_c_loss_sum = 0.0, 0.0, 0.0
            num_batches_in_epoch = 0

            for _ in range(steps_per_epoch):
                for _ in range(self.ci_discriminator_steps):
                    optimizerD.zero_grad()

                    noise_d = torch.randn(self.batch_size, self.random_dim, device=self.device)
                    cond_outputs_d = self.cond_generator.sample_train(self.batch_size)
                    c_d, _, col_d, opt_d = ( (torch.zeros((self.batch_size,0),device=self.device),None,None,None)
                                             if cond_outputs_d is None
                                             else (torch.from_numpy(cond_outputs_d[0]).to(self.device),
                                                   torch.from_numpy(cond_outputs_d[1]).to(self.device), 
                                                   cond_outputs_d[2], cond_outputs_d[3]) )

                    real_samples_np = data_sampler.sample(self.batch_size, col_d, opt_d)
                    real_samples = torch.from_numpy(real_samples_np.astype('float32')).to(self.device)

                    with torch.no_grad(): 
                        noise_cond_d = torch.cat([noise_d, c_d], dim=1).view(self.batch_size, -1, 1, 1)
                        fake_samples_img_d = self.generator(noise_cond_d)
                        fake_samples_flat_d = self.Gtransformer.inverse_transform(fake_samples_img_d)
                        fake_samples_activated_d = apply_activate(fake_samples_flat_d, self.transformer.output_info, self.gumbel_tau)

                    perm_indices = torch.randperm(self.batch_size, device=self.device)
                    c_d_perm_for_real = c_d[perm_indices] if c_d.numel() > 0 else c_d 

                    fake_input_d = self.Dtransformer.transform(torch.cat([fake_samples_activated_d, c_d], dim=1))
                    real_input_d = self.Dtransformer.transform(torch.cat([real_samples, c_d_perm_for_real], dim=1))

                    d_real_output, _ = discriminator(real_input_d)
                    loss_d_real = -torch.mean(d_real_output)
                    loss_d_real.backward()

                    d_fake_output, _ = discriminator(fake_input_d)
                    loss_d_fake = torch.mean(d_fake_output)
                    loss_d_fake.backward()

                    pen = calc_gradient_penalty_slerp(discriminator,
                                                      torch.cat([real_samples, c_d_perm_for_real], dim=1),
                                                      torch.cat([fake_samples_activated_d, c_d], dim=1),
                                                      self.Dtransformer, self.device, self.lambda_gp)
                    pen.backward()
                    optimizerD.step()
                    epoch_d_loss_sum += loss_d_real.item() + loss_d_fake.item() + pen.item()

                optimizerG.zero_grad()

                noise_g = torch.randn(self.batch_size, self.random_dim, device=self.device)
                cond_outputs_g = self.cond_generator.sample_train(self.batch_size) 
                c_g, m_g, _, _ = ( (torch.zeros((self.batch_size,0),device=self.device), torch.zeros((self.batch_size,0),device=self.device),None,None)
                                   if cond_outputs_g is None
                                   else (torch.from_numpy(cond_outputs_g[0]).to(self.device),
                                         torch.from_numpy(cond_outputs_g[1]).to(self.device), 
                                         cond_outputs_g[2], cond_outputs_g[3]) )

                noise_cond_g = torch.cat([noise_g, c_g], dim=1).view(self.batch_size, -1, 1, 1)

                fake_samples_g_img = self.generator(noise_cond_g)
                fake_samples_g_flat = self.Gtransformer.inverse_transform(fake_samples_g_img)
                fake_samples_g_activated = apply_activate(fake_samples_g_flat, self.transformer.output_info, self.gumbel_tau)

                fake_input_g_for_d = self.Dtransformer.transform(torch.cat([fake_samples_g_activated, c_g], dim=1))

                y_fake_for_g, info_fake_g = discriminator(fake_input_g_for_d)

                loss_g_adv = -torch.mean(y_fake_for_g)

                loss_g_cond = cond_loss(fake_samples_g_activated, self.transformer.output_info, c_g, m_g)

                with torch.no_grad(): 
                    _, info_real_g = discriminator(real_input_d)

                loss_mean = torch.norm(torch.mean(info_fake_g.view(self.batch_size, -1), dim=0) - torch.mean(info_real_g.view(self.batch_size, -1), dim=0), 1)
                loss_std = torch.norm(torch.std(info_fake_g.view(self.batch_size, -1), dim=0) - torch.std(info_real_g.view(self.batch_size, -1), dim=0), 1)
                loss_info = loss_mean + loss_std

                loss_g_aux = torch.tensor(0.0, device=self.device)
                if problem_type and classifier: 
                    fake_pred_for_g, fake_label_for_g = classifier(fake_samples_g_activated)

                    loss_fn_name_g = self._get_classifier_loss_fn_name(st_ed)
                    if loss_fn_name_g == 'regression':
                        current_c_loss_fn_g = SmoothL1Loss()
                        fake_label_for_g = fake_label_for_g.type_as(fake_pred_for_g).view_as(fake_pred_for_g)
                    elif loss_fn_name_g == 'binary':
                        current_c_loss_fn_g = BCELoss()
                        fake_label_for_g = fake_label_for_g.type_as(fake_pred_for_g)
                    else: 
                        current_c_loss_fn_g = CrossEntropyLoss()

                    if current_c_loss_fn_g:
                           loss_g_aux = current_c_loss_fn_g(fake_pred_for_g, fake_label_for_g)

                total_g_loss = (loss_g_adv +
                                 self.lambda_cond_loss * loss_g_cond +
                                 self.lambda_info_loss * loss_info +
                                 self.lambda_aux_classifier_loss * loss_g_aux)
                total_g_loss.backward()
                optimizerG.step()
                epoch_g_loss_sum += total_g_loss.item()

                if problem_type and classifier and optimizerC: 
                    optimizerC.zero_grad()
                    real_pred_c, real_label_c = classifier(real_samples.detach()) 

                    loss_fn_name_c = self._get_classifier_loss_fn_name(st_ed)
                    if loss_fn_name_c == 'regression':
                        current_c_loss_fn_c = SmoothL1Loss()
                        real_label_c = real_label_c.type_as(real_pred_c).view_as(real_pred_c)
                    elif loss_fn_name_c == 'binary':
                        current_c_loss_fn_c = BCELoss()
                        real_label_c = real_label_c.type_as(real_pred_c)
                    else: 
                        current_c_loss_fn_c = CrossEntropyLoss()

                    if current_c_loss_fn_c:
                        loss_c_on_real = current_c_loss_fn_c(real_pred_c, real_label_c)
                        loss_c_on_real.backward()
                        optimizerC.step()
                        epoch_c_loss_sum += loss_c_on_real.item()

                num_batches_in_epoch += 1

            avg_G_loss = epoch_g_loss_sum / num_batches_in_epoch if num_batches_in_epoch > 0 else float('inf')
            avg_D_loss = epoch_d_loss_sum / (num_batches_in_epoch * self.ci_discriminator_steps) if num_batches_in_epoch > 0 else float('inf')
            avg_C_loss = epoch_c_loss_sum / num_batches_in_epoch if problem_type and classifier and num_batches_in_epoch > 0 else 0.0

            self.G_losses.append(avg_G_loss); self.D_losses.append(avg_D_loss)
            if problem_type and classifier: self.C_losses.append(avg_C_loss)

            if trial:
                report_value = avg_G_loss
                trial.report(report_value, epoch_idx)
                if trial.should_prune():
                    if self.verbose: print(f"[Optuna Pruning] Trial 중단 at epoch {epoch_idx}")
                    raise optuna.exceptions.TrialPruned()

            if self.verbose and (epoch_idx + 1) % 10 == 0:
                log_msg = f"Epoch [{epoch_idx+1}/{self.epochs}], D Loss: {avg_D_loss:.4f}, G Loss: {avg_G_loss:.4f}"
                if problem_type and classifier and self.C_losses: log_msg += f", C Loss: {self.C_losses[-1]:.4f}"
                print(log_msg)
    def _get_classifier_loss_fn_name(self, st_ed_tuple):
        if not st_ed_tuple: return None
        target_dim_size = st_ed_tuple[1] - st_ed_tuple[0]
        if target_dim_size == 1: return 'regression'
        elif target_dim_size == 2: return 'binary'
        else: return 'multiclass'

    def sample(self, n_samples):
        self.generator.eval() 

        output_info_list = self.transformer.output_info 
        steps = n_samples // self.batch_size + 1
        generated_data_list = []

        with torch.no_grad():
            for _ in range(steps):
                noise_s = torch.randn(self.batch_size, self.random_dim, device=self.device)
                condvec_s = self.cond_generator.sample(self.batch_size)
                c_s = (torch.zeros((self.batch_size, 0), device=self.device)
                       if condvec_s is None or condvec_s.shape[1] == 0
                       else torch.from_numpy(condvec_s).to(self.device))

                noise_cond_s = torch.cat([noise_s, c_s], dim=1).view(self.batch_size, -1, 1, 1)

                fake_samples_img_s = self.generator(noise_cond_s)
                fake_samples_flat_s = self.Gtransformer.inverse_transform(fake_samples_img_s)
                fake_samples_activated_s = apply_activate(fake_samples_flat_s, output_info_list, self.gumbel_tau)
                generated_data_list.append(fake_samples_activated_s.cpu().numpy())

        generated_data_np = np.concatenate(generated_data_list, axis=0)

        result_data_np, num_to_resample = self.transformer.inverse_transform(generated_data_np)

        current_samples_list = list(result_data_np)

        resampling_attempts = 0 
        max_resampling_attempts = 5 

        while len(current_samples_list) < n_samples and resampling_attempts < max_resampling_attempts:
            if num_to_resample <= 0: 
                if self.verbose and len(current_samples_list) < n_samples:
                    print(f"Warning: 재샘플링 기준은 충족되었으나, 목표 샘플 수({n_samples})에 도달하지 못함 ({len(current_samples_list)}).")
                break 

            num_needed_additionally = n_samples - len(current_samples_list)
            num_to_generate_for_resample = min(num_to_resample, num_needed_additionally, self.batch_size)
            if num_to_generate_for_resample <=0: break 

            temp_resampled_list = []
            with torch.no_grad():
                noise_rs = torch.randn(num_to_generate_for_resample, self.random_dim, device=self.device)
                condvec_rs = self.cond_generator.sample(num_to_generate_for_resample)
                c_rs = (torch.zeros((num_to_generate_for_resample, 0), device=self.device)
                        if condvec_rs is None or condvec_rs.shape[1] == 0
                        else torch.from_numpy(condvec_rs).to(self.device))

                noise_cond_rs = torch.cat([noise_rs, c_rs], dim=1).view(num_to_generate_for_resample, -1, 1, 1)

                fake_resample_img = self.generator(noise_cond_rs)
                fake_resample_flat = self.Gtransformer.inverse_transform(fake_resample_img)
                fake_resample_activated = apply_activate(fake_resample_flat, output_info_list, self.gumbel_tau)
                temp_resampled_list.append(fake_resample_activated.cpu().numpy())

            if not temp_resampled_list: 
                if self.verbose: print("Warning: 이번 재샘플링 시도에서 데이터가 생성되지 않았습니다.")
                resampling_attempts += 1
                continue 

            data_resample_np = np.concatenate(temp_resampled_list, axis=0)

            res_new_data_np, num_to_resample_again = self.transformer.inverse_transform(data_resample_np)
            current_samples_list.extend(list(res_new_data_np))
            num_to_resample = num_to_resample_again 
            resampling_attempts += 1

            if self.verbose:
                print(f"재샘플링 시도 {resampling_attempts}: {len(current_samples_list)}/{n_samples} 샘플. 추가 재샘플링 필요 개수: {num_to_resample}.")

        final_result_data_np = np.array(current_samples_list)
        return final_result_data_np[:n_samples] 
