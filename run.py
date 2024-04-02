from os import makedirs
from subprocess import run

makedirs("results/multicore", exist_ok=True)

run(["make", "multicore"])

for N in [25000, 50000, 100000]:
    for n_threads in [1, 2, 4, 8, 16, 32]:
        runtimes = []
        for _ in range(5):
            process = run(
                ["./multicore", str(N), "10", "42"],
                capture_output=True,
                env={
                    "OMP_NUM_THREADS": str(n_threads)
                }
            )
            runtimes.append(process.stdout.decode("utf-8"))
        with open(f"results/multicore/{N}_bodies_{n_threads}_threads.txt", "a") as f:
            f.write("".join(runtimes))
