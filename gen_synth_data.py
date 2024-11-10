import argparse
import json
import os
import re
import math


def edit_a100_files(profile_dir):    
    # Iterate over all files in the profile directory
    for filename in os.listdir('./profile_data_samples'):
        # Match files with the pattern "DeviceType.A100_tpX_bsY.json"
        if re.match(r'DeviceType\.A100_tp\d+_bs\d+\.json', filename):
            # Construct full file path
            input_filepath = os.path.join('./profile_data_samples', filename)
            output_filepath = os.path.join(profile_dir, filename)
            
            # Load A100 JSON data
            with open(input_filepath, 'r') as file:
                a100_data = json.load(file)
            a100_data['execution_time']['between_forward_backward_time_ms'] = a100_data['execution_time']['forward_backward_time_ms'] \
                - sum(a100_data['execution_time']['layer_compute_total_ms'])
            a100_data['model']['num_MHA_layers'] = a100_data['model']['num_layers'] - 2
            with open(output_filepath, 'w') as outfile:
                json.dump(a100_data, outfile, indent=2)
            
            print(f"Generated: {output_filepath}")

def scale_up_tp_and_bs(profile_dir, final):
    # Iterate over all files in the profile directory
    for tp in (1,2,4):
        bs = 4
        while bs < final:
            # Match files with the pattern "DeviceType.A100_tpX_bsY.json"
            input_filepath = os.path.join(profile_dir, f'DeviceType.A100_tp{tp}_bs{bs}.json')
            output_filepath = os.path.join(profile_dir, f'DeviceType.A100_tp{tp}_bs{bs * 2}.json')
            # Load A100 JSON data
            with open(input_filepath, 'r') as file:
                input_data = json.load(file)
            input_data['execution_time']['layer_compute_total_ms'] = [
                    time * 2 for time in input_data['execution_time']['layer_compute_total_ms']
            ]
            input_data['execution_time']['forward_backward_time_ms'] *= 2 
            input_data['execution_time']['between_forward_backward_time_ms'] *= 2 
            input_data['execution_time']['total_time_ms'] = input_data['execution_time']['forward_backward_time_ms'] + \
                input_data['execution_time']['optimizer_time_ms'] + input_data['execution_time']['batch_generator_time_ms'] + \
                input_data['execution_time']['layernorm_grads_all_reduce_time_ms'] + input_data['execution_time']['embedding_grads_all_reduce_time_ms']
            
            with open(output_filepath, 'w') as outfile:
                json.dump(input_data, outfile, indent=2)
            bs *= 2
            print(f"Generated {output_filepath}")
    batch_sizes = [ 2**j for j in range(0,int(math.log2(final)+1))]
    for bs in batch_sizes:
        tp = 4
        while tp < final:
            # Match files with the pattern "DeviceType.A100_tpX_bsY.json"
            input_filepath = os.path.join(profile_dir, f'DeviceType.A100_tp{tp}_bs{bs}.json')
            output_filepath = os.path.join(profile_dir, f'DeviceType.A100_tp{tp*2}_bs{bs}.json')
            # Load A100 JSON data
            with open(input_filepath, 'r') as file:
                input_data = json.load(file)
            input_data['execution_time']['layer_compute_total_ms'] = [
                    time / 2 for time in input_data['execution_time']['layer_compute_total_ms']
            ]
            input_data['execution_time']['forward_backward_time_ms'] /= 2 
            input_data['execution_time']['between_forward_backward_time_ms'] /= 2 
            input_data['execution_time']['total_time_ms'] = input_data['execution_time']['forward_backward_time_ms'] + \
                input_data['execution_time']['optimizer_time_ms'] + input_data['execution_time']['batch_generator_time_ms'] + \
                input_data['execution_time']['layernorm_grads_all_reduce_time_ms'] + input_data['execution_time']['embedding_grads_all_reduce_time_ms']
            
            with open(output_filepath, 'w') as outfile:
                json.dump(input_data, outfile, indent=2)
            tp *= 2
            print(f"Generated {output_filepath}")
    

def scale_up_layers(profile_dir, layers):
    # Iterate over all files in the profile directory
    for filename in os.listdir(profile_dir):
        # Match files with the pattern "DeviceType.A100_tpX_bsY.json"
        if re.match(r'DeviceType\.A100_tp\d+_bs\d+\.json', filename):
            # Construct full file path
            a100_filepath = os.path.join(profile_dir, filename)
            # Load A100 JSON data
            with open(a100_filepath, 'r') as file:
                a100_data = json.load(file)
            # edit model config
            a100_data['model']['num_layers'] = layers
            a100_data['model']['num_MHA_layers'] = a100_data['model']['num_layers'] - 2
            parameters_per_layer = a100_data['model']["parameters"]["parameters_per_layer_bytes"]
            a100_data['model']["parameters"]["parameters_per_layer_bytes"] = [parameters_per_layer[0]] + \
                            [parameters_per_layer[1]] * (layers - 2) + \
                            [parameters_per_layer[-1]]
            a100_data['model']["parameters"]["total_parameters_bytes"] = sum(a100_data['model']["parameters"]["parameters_per_layer_bytes"])            
            
            # add the execution memory
            memory_per_layer = a100_data['execution_memory']["layer_memory_total_mb"]
            a100_data['execution_memory']["layer_memory_total_mb"] = [memory_per_layer[0]] + \
                            [memory_per_layer[1]] * (layers - 2) + \
                            [memory_per_layer[-1]]
            a100_data['execution_memory']["total_memory"] = sum(a100_data['execution_memory']["layer_memory_total_mb"]) 
            
            # edit execution time
            compute_per_layer = a100_data['execution_time']["layer_compute_total_ms"]
            a100_data['execution_time']["layer_compute_total_ms"] = [compute_per_layer[0]] + \
                            [compute_per_layer[1]] * (layers - 2) + \
                            [compute_per_layer[-1]]
            a100_data['execution_time']['forward_backward_time_ms'] = a100_data['execution_time']['between_forward_backward_time_ms'] +\
                sum(a100_data['execution_time']['layer_compute_total_ms'])
            exec = a100_data['execution_time']
            a100_data['execution_time']['layernorm_grads_all_reduce_time_ms'] = exec['layernorm_grads_all_reduce_time_ms'] / 10 * layers
            a100_data['execution_time']['optimizer_time_ms'] = exec['optimizer_time_ms'] / 10 * layers
            a100_data['execution_time']['total_time_ms'] = exec['forward_backward_time_ms'] +\
                exec['batch_generator_time_ms'] + exec['layernorm_grads_all_reduce_time_ms'] + \
                    exec['embedding_grads_all_reduce_time_ms']  + exec['optimizer_time_ms']
            
            with open(a100_filepath, 'w') as outfile:
                json.dump(a100_data, outfile, indent=2)
            
            print(f"Scaled the layer in {a100_filepath} from 10 to {layers}")
            
            
def create_v100_files(profile_dir):
    # Define the scaling factor for execution time
    factor = 3.5
    
    # Iterate over all files in the profile directory
    for filename in os.listdir(profile_dir):
        # Match files with the pattern "DeviceType.A100_tpX_bsY.json"
        if re.match(r'DeviceType\.A100_tp\d+_bs\d+\.json', filename):
            # Construct full file path
            a100_filepath = os.path.join(profile_dir, filename)
            
            # Load A100 JSON data
            with open(a100_filepath, 'r') as file:
                a100_data = json.load(file)
            
            # Prepare V100 data by scaling execution time fields
            v100_data = a100_data.copy()
            v100_data['execution_time']['total_time_ms'] *= factor
            v100_data['execution_time']['forward_backward_time_ms'] *= factor
            v100_data['execution_time']['batch_generator_time_ms'] *= factor
            v100_data['execution_time']['layernorm_grads_all_reduce_time_ms'] *= factor
            v100_data['execution_time']['embedding_grads_all_reduce_time_ms'] *= factor
            v100_data['execution_time']['optimizer_time_ms'] *= factor
            v100_data['execution_time']['layer_compute_total_ms'] = [
                time * factor for time in v100_data['execution_time']['layer_compute_total_ms']
            ]
            v100_data['execution_time']['between_forward_backward_time_ms'] *= factor
            
            # Generate the V100 filename and save the modified data
            v100_filename = filename.replace("A100", "V100")
            v100_filepath = os.path.join(profile_dir, v100_filename)
            
            with open(v100_filepath, 'w') as outfile:
                json.dump(v100_data, outfile, indent=2)
            
            print(f"Generated: {v100_filepath}")


def clear_profile_dir(profile_dir):
    # Iterate over all files in the profile directory
    for filename in os.listdir(profile_dir):
        file_path = os.path.join(profile_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic data files")
    parser.add_argument("layers", type=int, help="Number of layers")
    parser.add_argument("batch_size", type=int, help="Max batch size")
    parser.add_argument("--profile_dir", default="./profile", help="Profile directory")

    args = parser.parse_args()

    clear_profile_dir(args.profile_dir)
    edit_a100_files(args.profile_dir)
    scale_up_layers(args.profile_dir, args.layers)
    scale_up_tp_and_bs(args.profile_dir, args.batch_size)
    create_v100_files(args.profile_dir)

if __name__ == "__main__":
    main()