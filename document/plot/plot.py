import os, subprocess
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
import argparse

stat_sh = os.path.join(os.path.dirname(__file__), "stat.sh")

base_args = {
    "network": "garnet",
    "num-cpus": 64,
    "num-dirs": 64,
    "topology": "HyperX",
    "mesh-rows": 4,
    "routing-algorithm": 3,
    "dimwar-weight": "hop_x_cong",
    "enable-switch-collision-avoidance": True,
    "vcs-per-vnet": 16,
    "inj-vnet": 0,
    "synthetic": "uniform_random",
    "sim-cycles": 20000,
}


def build_cmd(outdir, **args) -> str:
    cmd = f"./build/NULL/gem5.opt --outdir={outdir} configs/example/garnet_synth_traffic.py"
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                cmd += f" --{k}"
        else:
            cmd += f" --{k}={v}"
    return cmd


def run_sim(injectionrate, override_args):
    args = {**base_args, **override_args}
    outdir = f"logs/report/{args['synthetic']}_vc{args['vcs-per-vnet']}/inj_{injectionrate:.3f}"
    os.makedirs(outdir, exist_ok=True)
    cmd = build_cmd(outdir, **args, injectionrate=injectionrate)
    with open(os.path.join(outdir, "cmd.txt"), "w") as f:
        f.write(cmd + "\n")
    subprocess.run(
        cmd,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    subprocess.run(
        ["bash", stat_sh, outdir], check=True, stdout=subprocess.PIPE
    )
    results = {}
    with open(os.path.join(outdir, "network_stats.txt")) as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            key, value = line.split("=")
            value = value.split(" (")[0]
            results[key.strip()] = float(value)
    return results


def run_exp(override_args):
    def run_once(injectionrate):
        return run_sim(injectionrate, override_args)

    with ThreadPoolExecutor(max_workers=8) as executor:
        injection_rates = [0.05 * i for i in range(1, 20)]
        results = list(executor.map(run_once, injection_rates))
    print(f"Finished {override_args}")
    return results


exps = {
    "DOR": run_exp({"routing-algorithm": 4}),
    "DimWar": run_exp({"routing-algorithm": 3}),
}

# Extract injection rates and latencies for plotting
injection_rates = [0.05 * i for i in range(1, 20)]

plt.figure(figsize=(10, 6))
for synthetic, results in exps.items():
    latencies = [result["average_packet_latency"] for result in results]
    plt.plot(injection_rates, latencies, marker="o", label=synthetic)

plt.xlabel("Injection Rate")
plt.ylabel("Average Packet Latency")
# plt.yscale('log')
plt.title("Average Packet Latency vs Injection Rate")
plt.legend()
plt.grid(True)
plt.savefig("document/plot/average_packet_latency_traffic.png")
