import json
import os
import re

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
            
            # Generate the V100 filename and save the modified data
            v100_filename = filename.replace("A100", "V100")
            v100_filepath = os.path.join(profile_dir, v100_filename)
            
            with open(v100_filepath, 'w') as outfile:
                json.dump(v100_data, outfile, indent=2)
            
            print(f"Generated: {v100_filepath}")

# Usage example
create_v100_files("./profile")