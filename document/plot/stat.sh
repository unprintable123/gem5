logging_target=$1/network_stats.txt
src_log=$1/stats.txt

echo > ${logging_target}
grep "packets_injected::total" $src_log | sed 's/system.ruby.network.packets_injected::total\s*/packets_injected = /' >> ${logging_target}
grep "packets_received::total" $src_log | sed 's/system.ruby.network.packets_received::total\s*/packets_received = /' >> ${logging_target}
grep "average_packet_queueing_latency" $src_log | sed 's/system.ruby.network.average_packet_queueing_latency\s*/average_packet_queueing_latency = /' >> ${logging_target}
grep "average_packet_network_latency" $src_log | sed 's/system.ruby.network.average_packet_network_latency\s*/average_packet_network_latency = /' >> ${logging_target}
grep "average_packet_latency" $src_log | sed 's/system.ruby.network.average_packet_latency\s*/average_packet_latency = /' >> ${logging_target}
grep "average_hops" $src_log | sed 's/system.ruby.network.average_hops\s*/average_hops = /' >> ${logging_target}
grep "reception_rate" $src_log | sed 's/system.ruby.network.reception_rate\s*/reception_rate = /' >> ${logging_target}
grep -E "system.ruby.network.link_utilization\S*out_r" $src_log | sed 's/system.ruby.network.link_utilization\(\S*\)\s*/link_utilization\1 = /' >> ${logging_target}
cat ${logging_target}
