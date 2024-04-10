from os import makedirs
from subprocess import run

makedirs("results/gpu", exist_ok=True)

for N in [25000, 50000, 100000]:
    for tpb in [32, 64, 128, 256, 512, 1024]:
        runtimes = []
        # nvcc -lm -use_fast_math -O3 gpu.cu -o gpu
        run([
            "nvcc",
            "-lm",
            "-use_fast_math",
            "-O3",
            "-D",
            f"BLOCKSIZE={tpb}",
            "gpu.cu",
            "-o",
            "gpu"
        ])
        
        for _ in range(5):

            process = run(
                ["./gpu", str(N), "10", "42"],
                capture_output=True,
            )
            runtimes.append(process.stdout.decode("utf-8"))
        
        with open(f"results/gpu/{N}_bodies_{tpb}_tpb.txt", "w") as f:
            f.write("".join(runtimes))