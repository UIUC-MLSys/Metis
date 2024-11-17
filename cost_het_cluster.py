# Copyright 2024 Samsung Electronics Co., Ltd. All Rights Reserved
import argparse
from typing import Dict, List, Tuple
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arguments import parse_args
from data_loader import ProfileDataLoader
from model.cost_estimator import HeteroCostEstimator
from model.activation_parameter import GPTActivationAndParam
from model.device_group import StagePerformance
from model.load_balancer import LayerLoadBalancer
from search_space.plan import IntraStagePlanGenerator, InterStagePlanGenerator
from gpu_cluster import GPUCluster
from utils import ModelConfig


def cost_het_cluster(args: argparse.Namespace, gpu_cluster: GPUCluster, profile_data: Dict, model_config: ModelConfig,
                     cost_estimator: HeteroCostEstimator, layer_load_balancer:LayerLoadBalancer, cache) -> List[Tuple]:

    estimate_costs = []
    inter_stage_costs = []
    inter_stage_plan_generator = InterStagePlanGenerator(device_types=set(gpu_cluster.get_device_types()),
                                                    num_devices=gpu_cluster.get_total_num_devices(),
                                                    gbs=args.gbs, num_layers=args.num_layers,
                                                    variance=args.min_group_scale_variance,
                                                    max_permute_len=args.max_permute_len)
    for inter_stage_plan in inter_stage_plan_generator:

        print(f'\n\ninter_stage_plan: {inter_stage_plan}')
        stage_performance = StagePerformance(model_config, profile_data, gpu_cluster, inter_stage_plan)
        rank_device_map = stage_performance.get_device_placement()

        intra_stage_plan_generator = IntraStagePlanGenerator(inter_stage_plan, stage_performance, layer_load_balancer, args.max_profiled_tp_degree, args.max_profiled_batch_size, cache)

        while intra_stage_plan_generator.has_next:
            intra_stage_plan = intra_stage_plan_generator.next()
            try:
                cost = cost_estimator.get_cost(inter_stage_plan, intra_stage_plan.strategies,
                                               intra_stage_plan.layer_partition, rank_device_map)
                print(f'cost: {cost}')
                estimate_costs.append((inter_stage_plan.node_sequence, inter_stage_plan.device_groups,
                                       intra_stage_plan.strategies, inter_stage_plan.batches,
                                       intra_stage_plan.layer_partition, intra_stage_plan.num_repartition, cost, intra_stage_plan.utility_measure))
            except KeyError as e:
                print(f'KeyError: {e}')
        inter_stage_costs.append(inter_stage_plan.costs)
    return estimate_costs, inter_stage_costs, cache


if __name__ == '__main__':
    args = parse_args()
    gpu_cluster = GPUCluster(hostfile_path=args.hostfile_path, clusterfile_path=args.clusterfile_path)

    data_loader = ProfileDataLoader(args.profile_data_path)
    profile_data, _ = data_loader.load_profile_data_all()
    print(profile_data)

    assert len(profile_data.keys()) > 0, 'There is no profiled data at the specified path.'

    model_config = ModelConfig(model_name=args.model_name, num_layers=args.num_layers,
                               sequence_length=args.sequence_length, vocab_size=args.vocab_size,
                               hidden_size=args.hidden_size, attention_head_size=args.attention_head_size)
    cache = {}
    model_volume = GPTActivationAndParam(model_config, profile_data['model']['parameters'])
    cost_estimator = HeteroCostEstimator(profile_data, model_config, model_volume, gpu_cluster)
    layer_load_balancer = LayerLoadBalancer(gpu_cluster, profile_data, model_config, args.gbs)

    start_time = time.time()
    estimate_costs, inter_stage_costs, cache = cost_het_cluster(args, gpu_cluster, profile_data, model_config, cost_estimator, layer_load_balancer, cache)
    end_time = time.time()
    sorted_result = sorted(estimate_costs, key=lambda kv: kv[6])
    print(f'len(costs): {len(estimate_costs)}')   
    print(
        'rank, cost, utility_measure, node_sequence, device_groups, strategies(dp_deg, tp_deg), batches(number of batch), layer_partition')
    costs = []
    for idx, result in enumerate(sorted_result):
        print(f'{idx + 1}, {result[6]}, {result[7]}, {result[0]}, {result[1]}, {result[2]}, {result[3]}, {result[4]}')
        costs.append(result[6])
    
    range_of_costs = []
    max_diff = 0
    min_diff = 99999
    max_index = 0
    min_index = 0
    for i in range(len(inter_stage_costs)):
        cost = inter_stage_costs[i]
        if len(cost) > 0:
            diff = max(cost) - min(cost)
            if diff > max_diff:
                max_diff = diff
                max_index = i
            elif diff < min_diff:
                min_diff = diff
                min_index = i
            range_of_costs.append(diff)
    print("max difference of costs: ", max(range_of_costs))
    print("min difference of costs: ", min(range_of_costs))
    max_stage = inter_stage_costs[max_index]
    min_stage = inter_stage_costs[min_index]
    print(f'Total time: {(end_time - start_time) * 1000} ms')
    print("cache =", cache)
    rank_min, rank_max = costs.index(min(max_stage)) + 1, costs.index(max(max_stage)) + 1
    print("Percentile of Min", ((rank_min) / len(costs)) * 100)
    print("Percentile of Max", ((rank_max) / len(costs)) * 100)
    rank_min, rank_max = costs.index(min(min_stage)) + 1, costs.index(max(min_stage)) + 1
    print("Percentile of Min", ((rank_min) / len(costs)) * 100)
    print("Percentile of Max", ((rank_max) / len(costs)) * 100)