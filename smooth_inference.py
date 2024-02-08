import torch
import time
import utils
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, choices=utils.DATASETS, default="imagenet", help="Dataset to use for benchmarking"
)
parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
parser.add_argument(
    "--precision", type=str, choices=["32-true", "16-mixed"], default="32-true", help="Precision for training"
)
parser.add_argument(
    "--devices",
    type=str,
    default="-1",
    help="Devices to use, see Lightning documentation. Comma separated list of idx. For single element use '0,'. -1 trains on all GPUs.",
)
parser.add_argument("--architecture", type=str, default="resnet50", help="Architecture to use for benchmarking")
parser.add_argument("--strategy", type=str, choices=["dp", "ddp"], default="ddp", help="Training strategy to use")
args = parser.parse_args()

cluster = utils.ClusterManager()


def main(args: argparse.Namespace):

    # get model and dataset
    fabric, model, optimizer, train_dataset, val_dataset, train_loader, val_loader = utils.prepare(
        args.dataset,
        args.architecture,
        args.batch_size,
        args.num_workers,
        args.devices,
        cluster,
        args.strategy,
        args.precision,
    )

    # testing loop
    loop_start_time = time.time()

    data_loading_times = []
    data_process_times = []
    forward_pass_times = []

    model.eval()
    fabric.barrier()
    end_of_batch_time = time.time()
    with torch.no_grad():
        for i in range(len(val_dataset)):
            inputs, _ = val_dataset[i]
            fabric.barrier()
            data_ready_time = time.time()
            data_loading_times.append(data_ready_time - end_of_batch_time)

            # preprocess data
            inputs = fabric.to_device(inputs)
            inputs = inputs.unsqueeze(0)  # Add batch dimension
            inputs = inputs.repeat(args.batch_size, 1, 1, 1)
            noise = torch.randn_like(inputs) * 0.25
            inputs = inputs + noise
            fabric.barrier()
            data_process_times.append(time.time() - data_ready_time)

            # forward pass
            forward_pass_start = time.time()
            outputs = model(inputs)
            outputs = fabric.all_gather(outputs)
            end_of_batch_time = time.time()

            forward_pass_times.append(end_of_batch_time - forward_pass_start)

            if i % 100 == 0 and i > 0:
                fabric.print(
                    f"Batch {i+1}: Data Loading Time: {data_loading_times[-1]:.4f} seconds, Forward Pass Time: {forward_pass_times[-1]:.4f} seconds. Batch Size: {len(inputs)}"
                )

    fabric.barrier()
    loop_time = time.time() - loop_start_time

    # print args
    utils.print_args(fabric, args)

    # print system details
    utils.print_system_info(fabric)

    # print benchmark results
    utils.print_benchmark_results(
        fabric, args, data_loading_times, data_loading_times, forward_pass_times, [0], loop_time
    )

    # return average times
    return np.mean(data_loading_times), np.mean(data_loading_times), np.mean(forward_pass_times), 0


if __name__ == "__main__":
    main(args)
