# Local
from YOLOSegPlusPlus import YOLOSegPlusPlus
from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from custom_yolo_trainer.custom_trainer import CustomSegmentationTrainer
from dataset import CustomDataset

# Internal Lib
import os
import time

# External Lib
import torch
from torch import nn
from torch.profiler import profile, ProfilerActivity, record_function

# Set environment variables to restrict other libraries to 1 thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Restrict PyTorch's intra-op parallelism to 1 thread
torch.set_num_threads(1)

# You can also check the current setting
print(f"PyTorch using {torch.get_num_threads()} threads.")

def benchmark_model(
    model,
    device="cpu",
    iterations=1000,
    warmup=100,
    batch=1,
):
    model.to(device)
    model.eval()

    dummy_data = torch.randn(batch, 4, 160, 160).to(device)

    with torch.no_grad():
        # Warm-up
        for _ in range(warmup):
            model.inference(dummy_data)

        # Timing
        start = time.perf_counter()
        for _ in range(iterations):
            model.inference(dummy_data)
        end = time.perf_counter()

    total_time = end - start
    avg_time_ms = (total_time / iterations) * 1000

    print(f"Total time: {total_time:.4f}s")
    print(f"Avg latency: {avg_time_ms:.3f} ms")

def profile_model(model: YOLOSegPlusPlus,
                  device: str = "gpu",
                  iterations: int = 1,
                  batch: int = 1,
                  do_warm_ups: bool = False,
                  warm_ups_i: int = 3,
                  ) -> None:
    model.to(device)
    model.eval()

    dummy_data = torch.randn(batch, 4, 160, 160).to(device)

    if device == "cpu":
        # Optional Warm-up
        if do_warm_ups:
            with torch.no_grad():
                for _ in range(warm_ups_i):
                    model.inference(dummy_data)

        with torch.no_grad():
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    for _ in range(iterations):
                        model.inference(dummy_data)

                        # ---YOLO Detect---
                        # model.yolo(dummy_data)
                        # ---YOLO Detect---

        # 3. Calculation
        # total_cpu_time is in microseconds (us)
        key_avg = prof.key_averages()

        # It's better to use 'cpu_time_total' from the record_function itself
        # or calculate via the sum of 'self_cpu_time_total'
        total_cpu_time_us = sum([item.self_cpu_time_total for item in key_avg])
        avg_time_ms = (total_cpu_time_us / iterations) / 1000

        print(f"\n--- Results over {iterations} iterations ---")
        print(f"Total CPU time: {total_cpu_time_us / 1e6:.4f} seconds")
        print(f"Average time per iteration: {avg_time_ms:.2f} ms")


#         # Get the key averages
#         key_avg = prof.key_averages()
#
#         # Calculate total CPU time from all operations
#         total_cpu_time = sum([item.self_cpu_time_total for item in key_avg])
#         avg_time_per_iteration = total_cpu_time / iterations
#
#         print(f"--- Results averaged over {iterations} iterations ---")
#         print(f"Total CPU time: {total_cpu_time / 1e6:.2f}")
#         print(f"Average time per iteration: {
#               avg_time_per_iteration / 1e6:.2f}")
#         print("\nPer-operation breakdown:")
#         print(prof.key_averages().table(
#             sort_by="cpu_time_total",
#             row_limit=10
#         ))

    # print(f"--- Results averaged over {iterations} iterations ---")
    # print(prof.key_averages().table(
    #     sort_by="cpu_time_total",
    #     row_limit=10
    # ))

    # else:
    #     with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    #         with record_function("model_inference"):
    #             model(dummy_data, dummy_logits)
    #     print(prof.key_averages().table(
    #         sort_by="cpu_time_total", row_limit=10))

    # ------KEEP FOR NOW------
    # print(
    #     prof.key_averages(group_by_input_shape=True).table(
    #         sort_by="cpu_time_total", row_limit=10
    #     )
    # )
    # print(prof.key_averages().table(
    #     sort_by="self_cpu_memory_usage", row_limit=10))
    # ------KEEP FOR NOW------

    else:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            device = "cuda"
            activities += [ProfilerActivity.CUDA]
        elif torch.xpu.is_available():
            device = "xpu"
            activities += [ProfilerActivity.XPU]
        else:
            print(
                "Neither CUDA nor XPU devices are available to demonstrate profiling on acceleration devices"
            )
            import sys

            sys.exit(0)

            sort_by_keyword = device + "_time_total"

            with profile(activities=activities, record_shapes=True) as prof:
                with record_function("model_inference"):
                    model(dummy_data)

                print(prof.key_averages().table(
                    sort_by=sort_by_keyword, row_limit=10))


if __name__ == "__main__":
    # Create predictor and load checkpoint
    p_args = dict(
        model="pretrained_detect_yolo/yolo12n_det_best/weights/best.pt",
        # model="pretrained_detect_yolo/yolo12n_det_coco/weights/best.pt",
        data=f"data/data.yaml",
        verbose=False,
        imgsz=160,
        save=False)

    # YOLO_predictor = CustomSegmentationPredictor(overrides=p_args)
    # YOLO_predictor.setup_model(p_args["model"])
    YOLO_trainer = CustomSegmentationTrainer(overrides=p_args)
    YOLO_trainer.setup_model()

    # Create model instance
    model = YOLOSegPlusPlus(predictor=YOLO_trainer,
                            training=False)

    # Profile model
    # profile_model(model=model,
    #               device="cpu",
    #               iterations=50,
    #               do_warm_ups=True,
    #               warm_ups_i=20,
    #               batch=1)

    benchmark_model(
        model,
        device="cpu",
        iterations=1000,
        warmup=100,
        batch=1,
    )
