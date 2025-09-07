```bash
scons build/NULL/gem5.opt PROTOCOL=Garnet_standalone -j $(nproc)
```





```bash
echo > network_stats.txt
grep "packets_injected::total" m5out/stats.txt | sed 's/system.ruby.network.packets_injected::total\s*/packets_injected = /' >> network_stats.txt
grep "packets_received::total" m5out/stats.txt | sed 's/system.ruby.network.packets_received::total\s*/packets_received = /' >> network_stats.txt
grep "average_packet_queueing_latency" m5out/stats.txt | sed 's/system.ruby.network.average_packet_queueing_latency\s*/average_packet_queueing_latency = /' >> network_stats.txt
grep "average_packet_network_latency" m5out/stats.txt | sed 's/system.ruby.network.average_packet_network_latency\s*/average_packet_network_latency = /' >> network_stats.txt
grep "average_packet_latency" m5out/stats.txt | sed 's/system.ruby.network.average_packet_latency\s*/average_packet_latency = /' >> network_stats.txt
grep "average_hops" m5out/stats.txt | sed 's/system.ruby.network.average_hops\s*/average_hops = /' >> network_stats.txt
grep "reception_rate" m5out/stats.txt | sed 's/system.ruby.network.reception_rate\s*/reception_rate = /' >> network_stats.txt
```

```bash
./build/NULL/gem5.opt configs/example/garnet_synth_traffic.py --network=garnet --num-cpus=64 --num-dirs=64 --topology=Mesh_XY --mesh-rows=8 --inj-vnet=0 --synthetic=uniform_random --sim-cycles=10000 --injectionrate=0.01 && bash document/stat.sh


./build/NULL/gem5.opt --debug-flags=RubyNetwork configs/example/garnet_synth_traffic.py --network=garnet --num-cpus=16 --num-dirs=16 --topology=Ring --inj-vnet=0 --synthetic=uniform_random --sim-cycles=10000 --injectionrate=0.5 --routing-algorithm=2 --vcs-per-vnet=1 --wormhole && bash document/stat.sh
./build/NULL/gem5.opt configs/example/garnet_synth_traffic.py --network=garnet --num-cpus=16 --num-dirs=16 --topology=Ring --inj-vnet=2 --synthetic=uniform_random --sim-cycles=10000 --injectionrate=0.01 --routing-algorithm=2 --vcs-per-vnet=1 --wormhole && bash document/stat.sh


./build/NULL/gem5.opt configs/example/garnet_synth_traffic.py --network=garnet --num-cpus=16 --num-dirs=16 --topology=Ring --inj-vnet=0 --synthetic=uniform_random --sim-cycles=10000 --injectionrate=0.01 && bash document/stat.sh
./build/NULL/gem5.opt configs/example/garnet_synth_traffic.py --network=garnet --num-cpus=16 --num-dirs=16 --topology=Ring --inj-vnet=0 --synthetic=uniform_random --sim-cycles=10000 --injectionrate=0.01 --routing-algorithm=2 && bash document/stat.sh
# --vcs-per-vnet
```
