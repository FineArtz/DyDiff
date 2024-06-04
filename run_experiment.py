import yaml
import argparse
import os
from time import sleep
from subprocess import Popen
import datetime
import dateutil

from rlkit.logger import config
from rlkit.logger.setup import build_nested_variant_generator


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", help="experiment specification file")
    args = parser.parse_args()

    args.nosrun = True  # Do not use srun

    with open(args.experiment, "r") as spec_file:
        spec_string = spec_file.read()
        exp_specs = yaml.safe_load(spec_string)

    # Check if GPU configuration is provided in the experiment specs and handle appropriately
    gpu_config = exp_specs["meta_data"].get("gpu", 0)  # Default to GPU 0 if not specified

    # generating the variants
    vg_fn = build_nested_variant_generator(exp_specs)

    # write all of them to a file
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime("%Y_%m_%d_%H_%M_%S")
    variants_dir = os.path.join(
        config.LOCAL_LOG_DIR,
        "variants-for-" + exp_specs["meta_data"]["exp_name"],
        "variants-" + timestamp,
    )
    os.makedirs(variants_dir)
    with open(os.path.join(variants_dir, "exp_spec_definition.yaml"), "w") as f:
        yaml.dump(exp_specs, f, default_flow_style=False)
    num_variants = 0
    for variant in vg_fn():
        i = num_variants
        variant["exp_id"] = i
        with open(os.path.join(variants_dir, "%d.yaml" % i), "w") as f:
            yaml.dump(variant, f, default_flow_style=False)
            f.flush()
        num_variants += 1

    num_workers = min(exp_specs["meta_data"]["num_workers"], num_variants)
    exp_specs["meta_data"]["num_workers"] = num_workers

    # Adjust command to distribute tasks across GPUs if gpu_config is a list
    running_processes = []
    args_idx = 0

    command = "python {script_path} -e {specs} -g {gpuid}"
    command_format_dict = exp_specs["meta_data"]

    if isinstance(gpu_config, list):  # If multiple GPUs are specified
        num_workers_per_gpu = max(1, num_workers // len(gpu_config))

    first_worker_launched = False

    while (args_idx < num_variants) or (len(running_processes) > 0):
        if (len(running_processes) < num_workers) and (args_idx < num_variants):
            if isinstance(gpu_config, list):  # Distribute tasks across GPUs
                gpu_id = gpu_config[args_idx % len(gpu_config)]
            else:
                gpu_id = gpu_config  # Single GPU ID

            if first_worker_launched:
                sleep(2)
            else:
                first_worker_launched = True  # 标记第一个worker已启动

            command_format_dict["specs"] = os.path.join(variants_dir, "%i.yaml" % args_idx)
            command_format_dict["gpuid"] = gpu_id
            command_to_run = command.format(**command_format_dict)
            command_to_run = command_to_run.split()
            print(command_to_run)
            p = Popen(command_to_run)
            sleep(1)
            args_idx += 1
            running_processes.append(p)
        else:
            sleep(1)

        new_running_processes = []
        for p in running_processes:
            ret_code = p.poll()
            if ret_code is None:
                new_running_processes.append(p)
        running_processes = new_running_processes