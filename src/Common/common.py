import gc, torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
	labels = pred.label_ids
	preds = pred.predictions.argmax(-1)
	precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average = "binary")
	acc = accuracy_score(labels, preds)

	return {
		'accuracy': acc,
		'f1': f1,
		'precision': precision,
		'recall': recall
	}


def print_gpu_utilization(idx: int = 0):
	nvmlInit()
	handle = nvmlDeviceGetHandleByIndex(idx)
	info = nvmlDeviceGetMemoryInfo(handle)
	print(f"\nGPU memory occupied: {info.used//1024**2} MB.")


def clear_cache(gpu_cnt: int = 1):
	
	gc.collect()

	for i in range(gpu_cnt):
		try:
			cuda_name = torch.cuda.get_device_name(i)
			print("\n/*====================================================================*/")
			print(f"device:{i}")
			print(cuda_name)
		except AssertionError:
			break

		print_gpu_utilization(i)

		torch.cuda.empty_cache()

		print()
		print(f"Allocated GPU Memory: {torch.cuda.memory_allocated(i)/(1024**3):.2f}GB")
		print(f"Reserved GPU Memory: {torch.cuda.memory_reserved(i)/(1024**3):.2f}GB")

	print("/*====================================================================*/")


class CommonObj:
	def __init__(self):
		pass
	def __del__(self):
		clear_cache(torch.cuda.device_count())

