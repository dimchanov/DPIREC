import torch
from torch import nn
import torch.nn.utils.prune as prune
import numpy as np


class Relevancer():
    def __init__(self, nn_model: nn.Module, d_channels_ratio: float, r_channels_drop_ratio: float, d_channels_drop_ratio: float = None) -> None:
        '''
        Args:
            nn_model (nn.Module): Neural network model.
            dchannels_ratio (float): The ratio of size of set of dummy parameters to size of set of r parameters.
            r_channels_drop_ratio (float): The ratio of real parameters that will be excluded at each iteration of training.
            d_channels_drop_ratio (float): The ratio of dummy parameters that will be excluded at each iteration of training.
        '''
        self.__nn_model = nn_model
        self.conv_modules = []
        for _, module in nn_model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.conv_modules.append(module)

        self.d_channels_ratio = d_channels_ratio
        self.r_channels_drop_ratio = r_channels_drop_ratio
        self.d_channels_drop_ratio = d_channels_drop_ratio
        if self.d_channels_drop_ratio is None:
            self.d_channels_drop_ratio = self.r_channels_drop_ratio

        self.criterion_values = []
        # r_channel
        self.r_channels_fixed_masks = []
        self.r_channels_tmp_masks = []
        self.r_channels_indexes = None
        self.r_channels_relevance_values = []
        # d_channels
        self.d_channels_masks = []
        self.d_channels_indexes = None
        self.d_channels_relevance_values = []


        self.init_channel_indexes()

    def init_channel_indexes(self):
        if self.r_channels_indexes is None:
            self.r_channels_indexes = []
            for conv_layer_index, module in enumerate(self.conv_modules):
                # r_channel
                r_channels_count = module.weight.shape[0]
                self.r_channels_fixed_masks.append(torch.ones(r_channels_count, dtype=torch.bool))
                self.r_channels_indexes.append(torch.arange(r_channels_count))

        self.r_channels_tmp_mask = []
        self.r_channels_relevance_values = []

        self.d_channels_indexes = []
        self.d_channels_relevance_values = []
        for conv_layer_index, module in enumerate(self.conv_modules):
            # r_channel
            r_channels_total_count = module.weight.shape[0]
            # self.r_channels_tmp_mask.append(torch.ones(r_channels_total_count, dtype=torch.bool))
            self.r_channels_relevance_values.append([[] for _ in range(r_channels_total_count)])
            # d_channels
            r_channels_curr_count = len(self.r_channels_indexes[conv_layer_index])
            d_channels_count = round(r_channels_curr_count * self.d_channels_ratio)
            self.d_channels_indexes.append(torch.arange(d_channels_count))
            self.d_channels_relevance_values.append([[] for _ in range(d_channels_count)])

    def update_channels_masks(self):
        # print("update_channels_masks: started")
        self.update_channels_relevance_values()
        self.delete_mask(True)
        for conv_layer_index, module in enumerate(self.conv_modules):
            # r_channel
            # количество исключаемых параметров
            r_channels_count = module.weight.shape[0]
            r_channels_drop_count = round(r_channels_count * self.r_channels_drop_ratio)
            if len(self.r_channels_indexes[conv_layer_index]) <= r_channels_drop_count:
                raise ValueError("It is impossible to exclude all channels of the neural network layer")
            r_channels_indexes_sample = random.sample(self.r_channels_indexes[conv_layer_index].tolist(), r_channels_drop_count)
            r_channels_mask = torch.ones(r_channels_count, dtype=torch.bool)
            r_channels_mask[r_channels_indexes_sample] = 0
            self.r_channels_tmp_masks.append(r_channels_mask)

            # d_channels
            # количество исключаемых параметров
            d_channels_count = round(module.weight.shape[0] * self.d_channels_ratio)
            d_channels_drop_count = round(d_channels_count * self.d_channels_drop_ratio)
            if len(self.d_channels_indexes[conv_layer_index]) <= d_channels_drop_count:
                raise ValueError("It is impossible to exclude all channels of the neural network layer")
            d_channels_indexes_sample = random.sample(self.d_channels_indexes[conv_layer_index].tolist(), d_channels_drop_count)
            d_channels_mask = torch.ones(d_channels_count, dtype=torch.bool)
            d_channels_mask[d_channels_indexes_sample] = 0
            self.d_channels_masks.append(d_channels_mask)
        # print("update_channels_masks: finished")
        self.apply_mask()

    def delete_mask(self, reassigning: bool = False):
        for conv_layer_index, module in enumerate(self.conv_modules):
            if not prune.is_pruned(module):
                continue

            with torch.no_grad():
                if hasattr(module, "weight"):
                    delattr(module, "weight")
                weight_orig = module._parameters["weight_orig"]
                del module._parameters["weight_orig"]
                del module._buffers["weight_mask"]
                module._forward_pre_hooks = OrderedDict()
                setattr(module, "weight", weight_orig)
        if reassigning:
            self.r_channels_tmp_masks = []
            self.d_channels_masks = []
        torch.cuda.empty_cache()

    def apply_mask(self):
        if not (self.r_channels_tmp_masks and len(self.r_channels_tmp_masks) == len(self.conv_modules)):
            print("Channels temp masks are empty")
            self.r_channels_tmp_masks = [[] for _ in range(len(self.conv_modules))]
            # return

        for conv_layer_index, module in enumerate(self.conv_modules):
            curr_layer_channels_tmp_masks = self.r_channels_tmp_masks[conv_layer_index]
            if len(curr_layer_channels_tmp_masks) != len(self.r_channels_fixed_masks[conv_layer_index]):
                curr_layer_channels_tmp_masks = torch.ones(len(self.r_channels_fixed_masks[conv_layer_index]), dtype=torch.bool)

            curr_layer_channels_tmp_masks &= self.r_channels_fixed_masks[conv_layer_index]
                
            curr_layer_weight_mask = torch.ones_like(module.weight, dtype=torch.bool)
            curr_layer_weight_mask[~curr_layer_channels_tmp_masks] = 0
            prune.custom_from_mask(module=module, name='weight', mask=curr_layer_weight_mask)

    def update_channels_relevance_values(self):
        if len(self.criterion_values) == 0:
            return
        mean_criterion_value = sum(self.criterion_values) / len(self.criterion_values)
        self.criterion_values = []
        for conv_layer_index, module in enumerate(self.conv_modules):
            r_channels_count = module.weight.shape[0]
            if len(self.r_channels_tmp_masks) <= conv_layer_index or len(self.r_channels_tmp_masks[conv_layer_index]) != r_channels_count:
                r_channels_mask = torch.ones(r_channels_count, dtype=torch.bool)
            else:
                r_channels_mask = self.r_channels_tmp_masks[conv_layer_index]
            r_channels_mask &= self.r_channels_fixed_masks[conv_layer_index]
            
            for r_channel_index in self.r_channels_indexes[conv_layer_index]:
                # print(r_channels_mask[r_channel_index], r_channel_index)
                if not r_channels_mask[r_channel_index]:
                    continue
                self.r_channels_relevance_values[conv_layer_index][r_channel_index].append(mean_criterion_value)

            d_channels_count = round(module.weight.shape[0] * self.d_channels_ratio)
            if len(self.d_channels_masks) <= conv_layer_index or len(self.d_channels_masks[conv_layer_index]) != d_channels_count:
                d_channels_mask = torch.ones(d_channels_count, dtype=torch.bool)
            else:
                d_channels_mask = self.d_channels_masks[conv_layer_index]
            for d_channel_index in self.d_channels_indexes[conv_layer_index]:
                if not d_channels_mask[d_channel_index]:
                    continue
                self.d_channels_relevance_values[conv_layer_index][d_channel_index].append(mean_criterion_value)

    def step(self, criterion_value: float):
        self.criterion_values.append(criterion_value)

    @staticmethod
    def get_target_relevance_values(channels_relevance_values: list, kind: str) -> list:
        target_relevance_values = []
        for channel_relevance_values in channels_relevance_values:
            if len(channel_relevance_values) == 0:
                continue
            
            target_relevance_value = channel_relevance_values[0]
            if kind == "value":
                relevance_values_sum = channel_relevance_values[0]
                for relevance_value_index in range(1, len(channel_relevance_values)):
                    target_relevance_value += (
                        channel_relevance_values[relevance_value_index] - relevance_values_sum / relevance_value_index)
                    relevance_values_sum += channel_relevance_values[relevance_value_index]
            elif kind == "straight" and len(channel_relevance_values) > 1:
                a, b = np.polyfit(np.arange(len(channel_relevance_values)), channel_relevance_values, deg=1)
                target_relevance_value = a

            target_relevance_values.append(target_relevance_value)
        return target_relevance_values

    def drop_by_probability(self, prune_ratio: int = None, delta: float = None, kind=None):
        print("drop_by_probability: started")
        self.update_channels_relevance_values()
        self.delete_mask(True)

        if prune_ratio is None and delta is None:
            prune_ratio = self.r_channels_drop_ratio
        if kind is None:
            kind = "value"
        elif kind not in ("value", "straight"):
            raise ValueError("drop_by_probability kind must be one of 'value' or 'straight'")

        for conv_layer_index, module in enumerate(self.conv_modules):
            d_target_relevance_values = self.get_target_relevance_values(self.d_channels_relevance_values[conv_layer_index], kind)
            mu = float(np.mean(d_target_relevance_values))
            sigma = float(np.std(d_target_relevance_values))
            r_target_relevance_values = self.get_target_relevance_values(self.r_channels_relevance_values[conv_layer_index], kind)

            if conv_layer_index == 1:
                print(r_target_relevance_values)

            r_channel_probabilities = [
                norm.cdf(r_trv, mu, sigma) for r_trv in r_target_relevance_values]

            if delta is not None:
                # prune_indexes = np.arange(len(probability_list))[np.array(probability_list) < delta]
                # print(f"prune_indexes: {len(prune_indexes)}")
                pass
            else:
                r_channels_drop_count = round(prune_ratio * module.weight.shape[0])
                r_channels_drop_indexes = self.r_channels_indexes[conv_layer_index][
                    np.argsort(r_channel_probabilities)[:r_channels_drop_count]]
                r_channels_save_indexes = self.r_channels_indexes[conv_layer_index][
                    np.argsort(r_channel_probabilities)[r_channels_drop_count:]]

            self.r_channels_indexes[conv_layer_index] = r_channels_save_indexes
        
            # curr_layer_channels_mask = torch.zeros(module.weight.shape[0], dtype=torch.bool)
            # curr_layer_channels_mask[curr_layer_channels_save_indexes] = 1
            # self.r_channels_tmp_masks.append(r_channels_mask)
            self.r_channels_fixed_masks[conv_layer_index][r_channels_drop_indexes] = 0

        self.apply_mask()
        # self.delete_mask()
        self.init_channel_indexes()
        print("drop_by_probability: finished")


