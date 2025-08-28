/*
 * Copyright (c) 2016 Georgia Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cpu/testers/garnet_synthetic_traffic/GarnetSyntheticTraffic.hh"

#include <cmath>
#include <iomanip>
#include <set>
#include <string>
#include <vector>

#include "base/logging.hh"
#include "base/random.hh"
#include "base/statistics.hh"
#include "debug/GarnetSyntheticTraffic.hh"
#include "mem/packet.hh"
#include "mem/port.hh"
#include "mem/request.hh"
#include "sim/sim_events.hh"
#include "sim/stats.hh"
#include "sim/system.hh"

namespace gem5
{

int TESTER_NETWORK=0;

bool
GarnetSyntheticTraffic::CpuPort::recvTimingResp(PacketPtr pkt)
{
    tester->completeRequest(pkt);
    return true;
}

void
GarnetSyntheticTraffic::CpuPort::recvReqRetry()
{
    tester->doRetry();
}

void
GarnetSyntheticTraffic::sendPkt(PacketPtr pkt)
{
    if (!cachePort.sendTimingReq(pkt)) {
        retryPkt = pkt; // RubyPort will retry sending
    }
    numPacketsSent++;
}

GarnetSyntheticTraffic::GarnetSyntheticTraffic(const Params &p)
    : ClockedObject(p),
      tickEvent([this]{ tick(); }, "GarnetSyntheticTraffic tick",
                false, Event::CPU_Tick_Pri),
      cachePort("GarnetSyntheticTraffic", this),
      retryPkt(NULL),
      size(p.memory_size),
      blockSizeBits(p.block_offset),
      numDestinations(p.num_dest),
      simCycles(p.sim_cycles),
      numPacketsMax(p.num_packets_max),
      numPacketsSent(0),
      singleSender(p.single_sender),
      singleDest(p.single_dest),
      trafficType(p.traffic_type),
      injRate(p.inj_rate),
      injVnet(p.inj_vnet),
      precision(p.precision),
      responseLimit(p.response_limit),
      requestorId(p.system->getRequestorId(this))
{
    // set up counters
    noResponseCycles = 0;
    schedule(tickEvent, 0);

    initTrafficType();
    if (trafficStringToEnum.count(trafficType) == 0) {
        fatal("Unknown Traffic Type: %s!\n", traffic);
    }
    traffic = trafficStringToEnum[trafficType];

    id = TESTER_NETWORK++;
    DPRINTF(GarnetSyntheticTraffic,"Config Created: Name = %s , and id = %d\n",
            name(), id);
}

Port &
GarnetSyntheticTraffic::getPort(const std::string &if_name, PortID idx)
{
    if (if_name == "test")
        return cachePort;
    else
        return ClockedObject::getPort(if_name, idx);
}

void
GarnetSyntheticTraffic::init()
{
    numPacketsSent = 0;
}


void
GarnetSyntheticTraffic::completeRequest(PacketPtr pkt)
{
    DPRINTF(GarnetSyntheticTraffic,
            "Completed injection of %s packet for address %x\n",
            pkt->isWrite() ? "write" : "read\n",
            pkt->req->getPaddr());

    assert(pkt->isResponse());
    noResponseCycles = 0;
    delete pkt;
}


void
GarnetSyntheticTraffic::tick()
{
    if (++noResponseCycles >= responseLimit) {
        fatal("%s deadlocked at cycle %d\n", name(), curTick());
    }

    // make new request based on injection rate
    // (injection rate's range depends on precision)
    // - generate a random number between 0 and 10^precision
    // - send pkt if this number is < injRate*(10^precision)
    bool sendAllowedThisCycle;
    double injRange = pow((double) 10, (double) precision);
    unsigned trySending = random_mt.random<unsigned>(0, (int) injRange);
    if (trySending < injRate*injRange)
        sendAllowedThisCycle = true;
    else
        sendAllowedThisCycle = false;

    // always generatePkt unless fixedPkts or singleSender is enabled
    if (sendAllowedThisCycle) {
        bool senderEnable = true;

        if (numPacketsMax >= 0 && numPacketsSent >= numPacketsMax)
            senderEnable = false;

        if (singleSender >= 0 && id != singleSender)
            senderEnable = false;

        if (senderEnable)
            generatePkt();
    }

    // Schedule wakeup
    if (curTick() >= simCycles)
        exitSimLoop("Network Tester completed simCycles");
    else {
        if (!tickEvent.scheduled())
            schedule(tickEvent, clockEdge(Cycles(1)));
    }
}

void
GarnetSyntheticTraffic::generatePkt()
{
    int num_destinations = numDestinations;

    // ---- Dimension inference (2D square or 3D cube only) ----
    // Try 2D: Rx * Ry, we assume square -> Rx = Ry = sqrt(N)
    auto is_perfect_square = [](int n) -> bool {
        int r = (int) std::round(std::sqrt((double)n));
        return r * r == n;
    };

    auto is_perfect_cube = [](int n) -> bool {
        int r = (int) std::round(std::cbrt((double)n));
        return r * r * r == n;
    };

    int Rx = -1, Ry = -1, Rz = -1;
    bool is2D = false, is3D = false;

    if (is_perfect_square(num_destinations)) {
        // 2D square
        Rx = (int) std::round(std::sqrt((double)num_destinations));
        Ry = Rx;
        Rz = 1;
        is2D = true;
    } else if (is_perfect_cube(num_destinations)) {
        // 3D cube
        Rx = (int) std::round(std::cbrt((double)num_destinations));
        Ry = Rx;
        Rz = Rx;
        is3D = true;
    } else {
        // Fallback: keep legacy assumption (2D square) to avoid crashes
        // but warn loudly. This keeps backward-compat if user passes non-square N.
        Rx = (int) std::round(std::sqrt((double)num_destinations));
        Ry = (Rx > 0) ? (num_destinations / Rx) : -1;
        Rz = 1;
        is2D = (Rx * Ry == num_destinations);
        if (!is2D) {
            fatal("Unsupported num_dest (%d): not perfect square/cube.\n",
                  num_destinations);
        }
    }

    // ---- source coordinate (x,y,z) from linear id ----
    unsigned destination = id;
    int source = id;

    int tmp = source;
    int src_x = tmp % Rx;
    tmp /= Rx;
    int src_y = tmp % Ry;
    tmp /= Ry;
    int src_z = tmp % Rz;

    // Destination coordinate to compute
    int dest_x = src_x;
    int dest_y = src_y;
    int dest_z = src_z;

    // Helper lambdas
    auto lin_id = [&](int x, int y, int z) -> int {
        // Convert (x,y,z) back to linear id
        return (z * Ry + y) * Rx + x;
    };

    auto rand_in = [&](int r) -> int {
        return (r > 1) ? random_mt.random<int>(0, r - 1) : 0;
    };

    auto bc_dim = [&](int coord, int R) -> int {
        // Bit-complement in a grid sense: farthest coordinate in that dimension
        return (R > 0) ? (R - 1 - coord) : 0;
    };


    // ---- Single-destination override ----
    if (singleDest >= 0) {
        destination = singleDest;
    }
    // ---- Existing legacy patterns ----
    else if (traffic == UNIFORM_RANDOM_) {
        destination = random_mt.random<unsigned>(0, num_destinations - 1);
    } else if (traffic == BIT_COMPLEMENT_) {
        // 2D legacy BC implemented for X/Y; extend to 3D by complementing all
        dest_x = bc_dim(src_x, Rx);
        dest_y = bc_dim(src_y, Ry);
        dest_z = bc_dim(src_z, Rz);
        destination = lin_id(dest_x, dest_y, dest_z);
    } else if (traffic == BIT_REVERSE_) {
        unsigned int straight = source;
        unsigned int reverse = source & 1; // LSB
        int num_bits = (int) std::log2(num_destinations);
        for (int i = 1; i < num_bits; i++) {
            reverse <<= 1;
            straight >>= 1;
            reverse |= (straight & 1); // LSB
        }
        destination = reverse;
    } else if (traffic == BIT_ROTATION_) {
        if (source % 2 == 0)
            destination = source / 2;
        else
            destination = ((source / 2) + (num_destinations / 2));
    } else if (traffic == NEIGHBOR_) {
        // Move +1 in X ring (2D/3D both OK; Z unchanged)
        dest_x = (src_x + 1) % Rx;
        destination = lin_id(dest_x, dest_y, dest_z);
    } else if (traffic == SHUFFLE_) {
        if (source < num_destinations / 2)
            destination = source * 2;
        else
            destination = (source * 2 - num_destinations + 1);
    } else if (traffic == TRANSPOSE_) {
        // 2D transpose: swap X<->Y; 3D: swap X<->Y, keep Z
        int tx = src_y;
        int ty = src_x;
        dest_x = (tx % Rx);
        dest_y = (ty % Ry);
        destination = lin_id(dest_x, dest_y, dest_z);
    } else if (traffic == TORNADO_) {
        // Standard tornado along X; Z unchanged
        int half = (int) std::ceil(Rx / 2.0) - 1;
        if (half < 0) half = 0;
        dest_x = (src_x + half) % Rx;
        destination = lin_id(dest_x, dest_y, dest_z);
    }
    // ---- New patterns: URB (X/Y/Z) ----
    else if (traffic == URB_X_) {
        // Target X with BC; others UR
        dest_x = bc_dim(src_x, Rx);
        dest_y = rand_in(Ry);
        dest_z = rand_in(Rz);
        destination = lin_id(dest_x, dest_y, dest_z);
    } else if (traffic == URB_Y_) {
        // Target Y with BC; others UR
        dest_x = rand_in(Rx);
        dest_y = bc_dim(src_y, Ry);
        dest_z = rand_in(Rz);
        destination = lin_id(dest_x, dest_y, dest_z);
    } else if (traffic == URB_Z_) {
        // Target Z with BC; others UR
        // If 2D (Rz==1), fall back to UR in 2D while keeping "intent" by BC on Y
        if (Rz == 1) {
            dest_x = rand_in(Rx);
            dest_y = bc_dim(src_y, Ry);
        } else {
            dest_x = rand_in(Rx);
            dest_y = rand_in(Ry);
            dest_z = bc_dim(src_z, Rz);
        }
        destination = lin_id(dest_x, dest_y, dest_z);
    }
    // ---- New pattern: S2 (Swap-2) ----
    else if (traffic == S2_) {
        // Even terminals use X dimension in BC-like way; odd use Y.
        // Other dimensions unchanged (Z unchanged for 3D).
        bool even = (source % 2 == 0);
        if (even) {
            dest_x = bc_dim(src_x, Rx); // adversarial along X
            dest_y = src_y;
        } else {
            dest_x = src_x;
            dest_y = bc_dim(src_y, Ry); // adversarial along Y
        }
        // Z remains src_z (unused bandwidth preserved)
        destination = lin_id(dest_x, dest_y, dest_z);
    }
    // ---- New pattern: DCR (Dimension Complement Reverse) ----
    else if (traffic == DCR_) {
        // For 3D: send to the "farthest" Z instance (complement Z plane),
        // and distribute across that plane uniformly in X,Y.
        // This stresses the Z bisection and models worst admissible traffic.
        if (Rz > 1) {
            dest_z = bc_dim(src_z, Rz);   // complement Z plane
            dest_x = rand_in(Rx);         // distribute across plane in X
            dest_y = rand_in(Ry);         // and Y
        } else {
            // 2D fallback: choose farthest Y row and distribute across X.
            dest_y = bc_dim(src_y, Ry);
            dest_x = rand_in(Rx);
        }
        destination = lin_id(dest_x, dest_y, dest_z);
    }
    else {
        fatal("Unknown Traffic Type: %s!\n", traffic);
    }

    // The source of the packets is a cache.
    // The destination of the packets is a directory.
    // The destination bits are embedded in the address after byte-offset.
    Addr paddr =  destination;
    paddr <<= blockSizeBits;
    unsigned access_size = 1; // Does not affect Ruby simulation

    // Modeling different coherence msg types over different msg classes.
    //
    // GarnetSyntheticTraffic assumes the Garnet_standalone coherence protocol
    // which models three message classes/virtual networks.
    // These are: request, forward, response.
    // requests and forwards are "control" packets (typically 8 bytes),
    // while responses are "data" packets (typically 72 bytes).
    //
    // Life of a packet from the tester into the network:
    // (1) This function generatePkt() generates packets of one of the
    //     following 3 types (randomly) : ReadReq, INST_FETCH, WriteReq
    // (2) mem/ruby/system/RubyPort.cc converts these to RubyRequestType_LD,
    //     RubyRequestType_IFETCH, RubyRequestType_ST respectively
    // (3) mem/ruby/system/Sequencer.cc sends these to the cache controllers
    //     in the coherence protocol.
    // (4) Network_test-cache.sm tags RubyRequestType:LD,
    //     RubyRequestType:IFETCH and RubyRequestType:ST as
    //     Request, Forward, and Response events respectively;
    //     and injects them into virtual networks 0, 1 and 2 respectively.
    //     It immediately calls back the sequencer.
    // (5) The packet traverses the network (simple/garnet) and reaches its
    //     destination (Directory), and network stats are updated.
    // (6) Network_test-dir.sm simply drops the packet.
    //
    MemCmd::Command requestType;

    RequestPtr req = nullptr;
    Request::Flags flags;

    // Inject in specific Vnet
    // Vnet 0 and 1 are for control packets (1-flit)
    // Vnet 2 is for data packets (5-flit)
    int injReqType = injVnet;

    if (injReqType < 0 || injReqType > 2)
    {
        // randomly inject in any vnet
        injReqType = random_mt.random(0, 2);
    }

    if (injReqType == 0) {
        // generate packet for virtual network 0
        requestType = MemCmd::ReadReq;
        req = std::make_shared<Request>(paddr, access_size, flags,
                                        requestorId);
    } else if (injReqType == 1) {
        // generate packet for virtual network 1
        requestType = MemCmd::ReadReq;
        flags.set(Request::INST_FETCH);
        req = std::make_shared<Request>(
            0x0, access_size, flags, requestorId, 0x0, 0);
        req->setPaddr(paddr);
    } else {  // if (injReqType == 2)
        // generate packet for virtual network 2
        requestType = MemCmd::WriteReq;
        req = std::make_shared<Request>(paddr, access_size, flags,
                                        requestorId);
    }

    req->setContext(id);

    //No need to do functional simulation
    //We just do timing simulation of the network

    DPRINTF(GarnetSyntheticTraffic,
            "Generated packet with destination %d, embedded in address %x\n",
            destination, req->getPaddr());

    PacketPtr pkt = new Packet(req, requestType);
    pkt->dataDynamic(new uint8_t[req->getSize()]);
    pkt->senderState = NULL;

    sendPkt(pkt);
}

void
GarnetSyntheticTraffic::initTrafficType()
{
    trafficStringToEnum["bit_complement"] = BIT_COMPLEMENT_;
    trafficStringToEnum["bit_reverse"] = BIT_REVERSE_;
    trafficStringToEnum["bit_rotation"] = BIT_ROTATION_;
    trafficStringToEnum["neighbor"] = NEIGHBOR_;
    trafficStringToEnum["shuffle"] = SHUFFLE_;
    trafficStringToEnum["tornado"] = TORNADO_;
    trafficStringToEnum["transpose"] = TRANSPOSE_;
    trafficStringToEnum["uniform_random"] = UNIFORM_RANDOM_;
    // Uniform Random Bisection variants
    trafficStringToEnum["urbx"] = URB_X_;
    trafficStringToEnum["urby"] = URB_Y_;
    trafficStringToEnum["urbz"] = URB_Z_;

    // Swap-2
    trafficStringToEnum["s2"] = S2_;

    // Dimension Complement Reverse
    trafficStringToEnum["dcr"] = DCR_;
}

void
GarnetSyntheticTraffic::doRetry()
{
    if (cachePort.sendTimingReq(retryPkt)) {
        retryPkt = NULL;
    }
}

void
GarnetSyntheticTraffic::printAddr(Addr a)
{
    cachePort.printAddr(a);
}

} // namespace gem5
