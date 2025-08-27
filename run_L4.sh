rm -rf m5out
scons build/NULL/gem5.opt -j$(nproc)

./build/NULL/gem5.opt \
configs/example/garnet_synth_traffic.py \
--network=garnet --num-cpus=64 --num-dirs=64 \
--topology=HyperX --mesh-rows=8 \
--routing-algorithm=3 \
--vcs-per-vnet=2 \
--inj-vnet=0 --synthetic=uniform_random \
--sim-cycles=10000 --injectionrate=0.01


logging_target="logs/trial0/HyperX_dimwar.txt"
mkdir -p "$(dirname "$logging_target")"
: > "$logging_target"

echo > ${logging_target}
grep "packets_injected::total" m5out/stats.txt | sed 's/system.ruby.network.packets_injected::total\s*/packets_injected = /' >> ${logging_target}
grep "packets_received::total" m5out/stats.txt | sed 's/system.ruby.network.packets_received::total\s*/packets_received = /' >> ${logging_target}
grep "average_packet_queueing_latency" m5out/stats.txt | sed 's/system.ruby.network.average_packet_queueing_latency\s*/average_packet_queueing_latency = /' >> ${logging_target}
grep "average_packet_network_latency" m5out/stats.txt | sed 's/system.ruby.network.average_packet_network_latency\s*/average_packet_network_latency = /' >> ${logging_target}
grep "average_packet_latency" m5out/stats.txt | sed 's/system.ruby.network.average_packet_latency\s*/average_packet_latency = /' >> ${logging_target}
grep "average_hops" m5out/stats.txt | sed 's/system.ruby.network.average_hops\s*/average_hops = /' >> ${logging_target}
grep "reception_rate" m5out/stats.txt | sed 's/system.ruby.network.reception_rate\s*/reception_rate = /' >> ${logging_target}
cat ${logging_target}
