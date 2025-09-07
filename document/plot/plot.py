import os, subprocess
from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt
import argparse

stat_sh = os.path.join(os.path.dirname(__file__), "stat.sh")

base_args = {
    "network": "garnet",
    "num-cpus": 64,
    "num-dirs": 64,
    "topology": "Mesh_XY",
    "routing-algorithm": 1,
    "mesh-rows": 8,
    "inj-vnet": 0,
    "synthetic": "uniform_random",
    "sim-cycles": 10000,
    "vcs-per-vnet": 16,
}


def build_cmd(outdir, **args) -> str:
    cmd = f"./build/NULL/gem5.opt --outdir={outdir} configs/example/garnet_synth_traffic.py"
    for k, v in args.items():
        cmd += f" --{k}={v}"
    return cmd


def run_once(cmd, outdir):
    if os.path.exists(os.path.join(outdir, "cmd.txt")):
        old_cmd = open(os.path.join(outdir, "cmd.txt")).read().strip()
        if old_cmd == cmd.strip():
            return
        else:
            print(cmd)
            print(old_cmd)
            raise ValueError(
                f"Output dir {outdir} already exists with different command."
            )
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


def run_sim(injectionrate, override_args):
    args = {**base_args, **override_args}
    appendix = ""
    if "router-latency" in args:
        appendix += f"_rl{args['router-latency']}"
    if "link-width-bits" in args:
        appendix += f"_lw{args['link-width-bits']}"
    outdir = f"logs/lab2/{args['synthetic']}_vc{args['vcs-per-vnet']}{appendix}/inj_{injectionrate:.3f}"
    os.makedirs(outdir, exist_ok=True)
    cmd = build_cmd(outdir, **args, injectionrate=injectionrate)
    run_once(cmd, outdir)
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

    with ThreadPoolExecutor(max_workers=12) as executor:
        injection_rates = [0.02 * i for i in range(1, 45)]
        results = list(executor.map(run_once, injection_rates))
    print(f"Finished {override_args}")
    return results


exps = {f"VC{vc}": run_exp({"vcs-per-vnet": vc}) for vc in [1, 2, 4, 8, 16]}

# Extract injection rates and latencies for plotting
injection_rates = [0.02 * i for i in range(1, 45)]
markers = ["o", "s", "^", "D", "v", "P", "*"]

plt.figure(figsize=(10, 6))
for i, (synthetic, results) in enumerate(exps.items()):
    receptions = [result["reception_rate"] for result in results]
    latencies = [result["average_packet_latency"] for result in results]
    receptions, latencies = zip(*sorted(zip(receptions, latencies)))
    plt.plot(receptions, latencies, marker=markers[i], label=synthetic)

plt.xlabel("Reception Rate")
plt.ylabel("Average Packet Latency")
plt.ylim(12, 400)
# plt.yscale('log')
plt.title("Latency-Throughput Curve for Different Traffic Patterns")
plt.legend()
plt.grid(True)
plt.savefig("document/plot/average_packet_latency_traffic.png")
