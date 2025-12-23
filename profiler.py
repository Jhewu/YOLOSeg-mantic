# Local
from YOLOSegPlusPlus import YOLOSegPlusPlus
from custom_yolo_predictor.custom_detseg_predictor import CustomSegmentationPredictor
from dataset import CustomDataset

# Internal Lib
import os

# External Lib
import torch
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


def get_logits(model: YOLOSegPlusPlus,
               x: torch.tensor,
               device: str = "cuda"):
    model = model.predictor.model.model
    model.to(device)

    x = model(x)

    detect_branch, cls_branch = x
    a, b, c = cls_branch

    return a[:, -1:]


def profile_model(model: YOLOSegPlusPlus,
                  device: str = "gpu",
                  calculate_logits: bool = False,
                  ) -> None:
    model.to(device)
    dummy_data = torch.randn(128, 4, 160, 160).to(device)
    dummy_logits = torch.randn(128, 1, 20, 20).to(device)

    if device == "cpu":
        if calculate_logits:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    dummy_logits = get_logits(model, dummy_data, device)
                    model(dummy_data, dummy_logits)
                    print(prof.key_averages().table(
                        sort_by="cpu_time_total", row_limit=10))
        else:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
                with record_function("model_inference"):
                    model(dummy_data, dummy_logits)
            print(prof.key_averages().table(
                sort_by="cpu_time_total", row_limit=10))

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
                model(inputs)

        print(prof.key_averages().table(sort_by=sort_by_keyword, row_limit=10))


if __name__ == "__main__":
    # Create predictor and load checkpoint
    p_args = dict(model="pretrained_detect_yolo/best_yolo12n_det/weights/best.pt",
                  data=f"data/data.yaml",
                  verbose=True,
                  imgsz=160,
                  save=False)

    YOLO_predictor = CustomSegmentationPredictor(overrides=p_args)
    YOLO_predictor.setup_model(p_args["model"])

    # Create model instance
    model = YOLOSegPlusPlus(predictor=YOLO_predictor)

    # Profile model
    profile_model(model=model,
                  device="cpu",
                  calculate_logits=True)
