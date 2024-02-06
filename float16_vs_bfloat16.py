import torch

def find_mismatch(start, incr):
    bf16 = torch.tensor(start, dtype=torch.bfloat16)
    print(f"\nfp32 start={start:.2e} using increment={incr}")
    print(f"{'bfloat16':>18} {'float16':>18} {'diff':>8}")
    c = 0
    tries = 0
    while c<8:
        fp16 = bf16.to(torch.float16)
        if not(fp16 == bf16):
            print(f"{bf16:.16f} {fp16:.16f} {torch.sub(fp16.to(dtype=torch.float32), bf16):+.2e}")
            c += 1
        bf16 += incr
        tries += 1
        if tries >= 1e5:
            print(f"gave up finding mismatch after {tries} steps")
            return
        
find_mismatch(1e-08, 1e-09)
find_mismatch(1e-07, 1e-08)
