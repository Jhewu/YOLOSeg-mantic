# Local
from YOLOSegPlusPlus import YOLOSegPlusPlus
from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from custom_yolo_trainer.custom_trainer import CustomSegmentationTrainer
from dataset import CustomDataset

# Internal Lib
import os

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


def profile_model(model: YOLOSegPlusPlus,
                  device: str = "gpu",
                  iterations: int = 1,
                  batch: int = 1,
                  warm_ups: bool = False,
                  ) -> None:
    model.to(device)
    model.eval()

    dummy_data = torch.randn(batch, 4, 160, 160).to(device)

    if device == "cpu":
        # Optional Warm-up
        if warm_ups:
            for _ in range(2):
                model.inference(dummy_data)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            with record_function("model_inference"):
                for _ in range(iterations):
                    model.inference(dummy_data)

                    # ---YOLO Detect---
                    # model.yolo(dummy_data)
                    # ---YOLO Detect---

        # Get the key averages
        key_avg = prof.key_averages()

        # Calculate total CPU time from all operations
        total_cpu_time = sum([item.self_cpu_time_total for item in key_avg])
        avg_time_per_iteration = total_cpu_time / iterations

        print(f"--- Results averaged over {iterations} iterations ---")
        print(f"Total CPU time: {total_cpu_time / 1e6:.2f}")
        print(f"Average time per iteration: {
              avg_time_per_iteration / 1e6:.2f}")
        print("\nPer-operation breakdown:")
        print(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=10
        ))

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
    profile_model(model=model,
                  device="cpu",
                  iterations=3,
                  warm_ups=True,
                  batch=1)
