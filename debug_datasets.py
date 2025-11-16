from datasets import load_dataset

# Check STS-B structure
print("=== STS-B Dataset Structure ===")
ds = load_dataset("sentence-transformers/stsb")
print("Available splits:", list(ds.keys()))
print("Train columns:", ds['train'].column_names)
print("Sample from train:", ds['train'][0])

print("\n=== SimLex Dataset Structure ===")
try:
    ds_simlex = load_dataset("tasksource/simlex")
    print("Available splits:", list(ds_simlex.keys()))
    print("Train columns:", ds_simlex['train'].column_names)
    print("Sample from train:", ds_simlex['train'][0])
except Exception as e:
    print("Error loading SimLex:", e)

print("\n=== STS12 Dataset Structure ===")
try:
    ds_sts12 = load_dataset("mteb/sts12-sts")
    print("Available splits:", list(ds_sts12.keys()))
    print("Test columns:", ds_sts12['test'].column_names)
    print("Sample from test:", ds_sts12['test'][0])
except Exception as e:
    print("Error loading STS12:", e)
