/*
 * Copyright (c) 2008 Princeton University
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


#include "mem/ruby/network/garnet/RoutingUnit.hh"

#include "base/cast.hh"
#include "base/compiler.hh"
#include "debug/RubyNetwork.hh"
#include "mem/ruby/network/garnet/InputUnit.hh"
#include "mem/ruby/network/garnet/OutputUnit.hh"
#include "mem/ruby/network/garnet/Router.hh"
#include "mem/ruby/slicc_interface/Message.hh"

#include <cstdlib>    // getenv
#include <sstream>
#include <cstring>
#include <cctype>

namespace gem5
{

namespace ruby
{

namespace garnet
{

RoutingUnit::RoutingUnit(Router *router)
{
    m_router = router;
    m_routing_table.clear();
    m_weight_table.clear();
}

void
RoutingUnit::addRoute(std::vector<NetDest>& routing_table_entry)
{
    if (routing_table_entry.size() > m_routing_table.size()) {
        m_routing_table.resize(routing_table_entry.size());
    }
    for (int v = 0; v < routing_table_entry.size(); v++) {
        m_routing_table[v].push_back(routing_table_entry[v]);
    }
}

void
RoutingUnit::addWeight(int link_weight)
{
    m_weight_table.push_back(link_weight);
}

bool
RoutingUnit::supportsVnet(int vnet, std::vector<int> sVnets)
{
    // If all vnets are supported, return true
    if (sVnets.size() == 0) {
        return true;
    }

    // Find the vnet in the vector, return true
    if (std::find(sVnets.begin(), sVnets.end(), vnet) != sVnets.end()) {
        return true;
    }

    // Not supported vnet
    return false;
}

/*
 * This is the default routing algorithm in garnet.
 * The routing table is populated during topology creation.
 * Routes can be biased via weight assignments in the topology file.
 * Correct weight assignments are critical to provide deadlock avoidance.
 */
int
RoutingUnit::lookupRoutingTable(int vnet, NetDest msg_destination)
{
    // First find all possible output link candidates
    // For ordered vnet, just choose the first
    // (to make sure different packets don't choose different routes)
    // For unordered vnet, randomly choose any of the links
    // To have a strict ordering between links, they should be given
    // different weights in the topology file

    int output_link = -1;
    int min_weight = INFINITE_;
    std::vector<int> output_link_candidates;
    int num_candidates = 0;

    // Identify the minimum weight among the candidate output links
    for (int link = 0; link < m_routing_table[vnet].size(); link++) {
        if (msg_destination.intersectionIsNotEmpty(
            m_routing_table[vnet][link])) {

        if (m_weight_table[link] <= min_weight)
            min_weight = m_weight_table[link];
        }
    }

    // Collect all candidate output links with this minimum weight
    for (int link = 0; link < m_routing_table[vnet].size(); link++) {
        if (msg_destination.intersectionIsNotEmpty(
            m_routing_table[vnet][link])) {

            if (m_weight_table[link] == min_weight) {
                num_candidates++;
                output_link_candidates.push_back(link);
            }
        }
    }

    if (output_link_candidates.size() == 0) {
        fatal("Fatal Error:: No Route exists from this Router.");
        exit(0);
    }

    // Randomly select any candidate output link
    int candidate = 0;
    if (!(m_router->get_net_ptr())->isVNetOrdered(vnet))
        candidate = rand() % num_candidates;

    output_link = output_link_candidates.at(candidate);
    return output_link;
}


void
RoutingUnit::addInDirection(PortDirection inport_dirn, int inport_idx)
{
    m_inports_dirn2idx[inport_dirn] = inport_idx;
    m_inports_idx2dirn[inport_idx]  = inport_dirn;
}

void
RoutingUnit::addOutDirection(PortDirection outport_dirn, int outport_idx)
{
    m_outports_dirn2idx[outport_dirn] = outport_idx;
    m_outports_idx2dirn[outport_idx]  = outport_dirn;
}

// outportCompute() is called by the InputUnit
// It calls the routing table by default.
// A template for adaptive topology-specific routing algorithm
// implementations using port directions rather than a static routing
// table is provided here.

int
RoutingUnit::outportCompute(RouteInfo route, int inport,
                            PortDirection inport_dirn)
{
    int outport = -1;

    if (route.dest_router == m_router->get_id()) {

        // Multiple NIs may be connected to this router,
        // all with output port direction = "Local"
        // Get exact outport id from table
        outport = lookupRoutingTable(route.vnet, route.net_dest);
        return outport;
    }

    // Routing Algorithm set in GarnetNetwork.py
    // Can be over-ridden from command line using --routing-algorithm = 1
    RoutingAlgorithm routing_algorithm =
        (RoutingAlgorithm) m_router->get_net_ptr()->getRoutingAlgorithm();

    switch (routing_algorithm) {
        case TABLE_:  outport =
            lookupRoutingTable(route.vnet, route.net_dest); break;
        case XY_:     outport =
            outportComputeXY(route, inport, inport_dirn); break;
        // any custom algorithm
        case CUSTOM_: outport =
            outportComputeCustom(route, inport, inport_dirn); break;
        case DIMWAR_: outport =
            outportComputeDimWar(route, inport, inport_dirn); break;
        default: outport =
            lookupRoutingTable(route.vnet, route.net_dest); break;
    }

    assert(outport != -1);
    return outport;
}

// XY routing implemented using port directions
// Only for reference purpose in a Mesh
// By default Garnet uses the routing table
int
RoutingUnit::outportComputeXY(RouteInfo route,
                              int inport,
                              PortDirection inport_dirn)
{
    PortDirection outport_dirn = "Unknown";

    [[maybe_unused]] int num_rows = m_router->get_net_ptr()->getNumRows();
    int num_cols = m_router->get_net_ptr()->getNumCols();
    assert(num_rows > 0 && num_cols > 0);

    int my_id = m_router->get_id();
    int my_x = my_id % num_cols;
    int my_y = my_id / num_cols;

    int dest_id = route.dest_router;
    int dest_x = dest_id % num_cols;
    int dest_y = dest_id / num_cols;

    int x_hops = abs(dest_x - my_x);
    int y_hops = abs(dest_y - my_y);

    bool x_dirn = (dest_x >= my_x);
    bool y_dirn = (dest_y >= my_y);

    // already checked that in outportCompute() function
    assert(!(x_hops == 0 && y_hops == 0));

    if (x_hops > 0) {
        if (x_dirn) {
            assert(inport_dirn == "Local" || inport_dirn == "West");
            outport_dirn = "East";
        } else {
            assert(inport_dirn == "Local" || inport_dirn == "East");
            outport_dirn = "West";
        }
    } else if (y_hops > 0) {
        if (y_dirn) {
            // "Local" or "South" or "West" or "East"
            assert(inport_dirn != "North");
            outport_dirn = "North";
        } else {
            // "Local" or "North" or "West" or "East"
            assert(inport_dirn != "South");
            outport_dirn = "South";
        }
    } else {
        // x_hops == 0 and y_hops == 0
        // this is not possible
        // already checked that in outportCompute() function
        panic("x_hops == y_hops == 0");
    }

    return m_outports_dirn2idx[outport_dirn];
}

// Template for implementing custom routing algorithm
// using port directions. (Example adaptive)
int
RoutingUnit::outportComputeCustom(RouteInfo route,
                                 int inport,
                                 PortDirection inport_dirn)
{
    panic("%s placeholder executed", __FUNCTION__);
}

int
RoutingUnit::outportComputeDimWar(RouteInfo route, int inport,
                                    PortDirection inport_dirn)
{
    const int num_rows = m_router->get_net_ptr()->getNumRows();
    const int num_cols = m_router->get_net_ptr()->getNumCols();
    const int my = m_router->get_id();
    const int my_x = my % num_cols, my_y = my / num_cols;
    const int dst = route.dest_router;
    const int dst_x = dst % num_cols, dst_y = dst / num_cols;

    // If we are at the destination router, use the routing table
    if (dst == my)
    {
        return lookupRoutingTable(route.vnet, route.net_dest);
    }

    // Determine the dimension we are going to route in this hop
    bool x_unaligned = (dst_x != my_x);
    bool y_unaligned = (dst_y != my_y);
    int dim = x_unaligned ? 0 : (y_unaligned ? 1 : -1);
    assert(dim != -1);

    bool allow_deroute =
        (m_router->getInputUnit(inport)->getLastHeadInClass() == 0);

    std::vector<int> candidates;
    std::vector<int> rem_hops;
    std::vector<int> classes;

    for (const auto &[dirn, idx] : m_outports_dirn2idx)
    {
        const std::string &s = dirn;
        if (s.rfind("out_r", 0) != 0)
            continue;
        int neigh = std::stoi(s.substr(5));
        int nx = neigh % num_cols, ny = neigh / num_cols;

        // Check if this output port is in the right dimension
        if (dim == 0)
        {
            if (ny != my_y)
                continue;
        }
        else
        { // dim == 1
            if (nx != my_x)
                continue;
        }

        // Check if this is a minimal route
        bool is_minimal = (dim == 0) ? (nx == dst_x) : (ny == dst_y);

        if (is_minimal)
        {
            // minimal route
            candidates.push_back(idx);

            int hops_left = 1 + ((dim == 0 && y_unaligned) ? 1 : 0);
            rem_hops.push_back(hops_left);
            classes.push_back(0); // VC0
        }
        else if (allow_deroute)
        {
            // deroute is allowed
            // (we are in class 1 if we deroute)
            candidates.push_back(idx);

            int hops_left = 1 + 1 + ((dim == 0 && y_unaligned) ? 1 : 0);
            rem_hops.push_back(hops_left);
            classes.push_back(1); // VC1
        }
    }

    // If no candidates found, fall back to routing table
    if (candidates.empty())
    {
        fatal("DimWar: no candidates found");
    }

    if (!allow_deroute) {
        assert(candidates.size() == 1);
        m_router->getInputUnit(inport)->setPendingRouteClass(0);
        return candidates[0];
    }

    assert(candidates.size() == (dim ? num_rows - 1 : num_cols - 1));

    double best_w = 1e100;
    int best = -1;
    int best_cls = 0;
    for (int i = 0; i < (int)candidates.size(); ++i)
    {
        int i_rr = (i + m_rr_idx[dim]) % candidates.size();
        double w = dimwarWeight(candidates[i_rr], route.vnet, rem_hops[i_rr]);
        if (w < best_w - 1e-9) // avoid floating-point tie
        {
            best_w = w;
            best = candidates[i_rr];
            best_cls = classes[i_rr];
        }
    }

    // Remember the route class for the next hop
    m_router->getInputUnit(inport)->setPendingRouteClass(best_cls);
    // Update the round-robin index for next time
    m_rr_idx[dim] = (m_rr_idx[dim] + 1) % candidates.size();

    assert(best != -1);
    return best;
}


double
RoutingUnit::dimwarWeight(int outport_idx, int vnet, int remaining_hops)
{
    auto* out = m_router->getOutputUnit(outport_idx);

    auto* net = m_router->get_net_ptr();
    const int    mode  = net->getDimWarWeightMode();
    const double A     = net->getDimWarAlpha();
    const double B     = net->getDimWarBeta();
    const double G     = net->getDimWarGamma();

    // cong1 = 1/(1+free_vcs), cong2 = 1/(1+sum_credits)
    int free_vcs = out->num_free_vcs(vnet);
    int sum_creds = out->sum_credits(vnet);
    double cong1 = 1.0 / (1.0 + (double)free_vcs);
    double cong2 = 1.0 / (1.0 + (double)sum_creds);
    double hops  = (double)remaining_hops;

    switch (mode) {
    case 0: // hop_x_cong
        return cong1 * hops;
    case 1: // linear: alpha*hops + beta*cong1
        return A * hops + B * cong1;
    case 2: // credits: alpha*hops + beta*cong2
        return A * hops + B * cong2;
    case 3: // hop only
        return hops;
    case 4: // cong only
        return cong1;
    case 5: // hybrid: alpha*hops + beta*cong1 + gamma*cong2
        return A * hops + B * cong1 + G * cong2;
    default:
        return cong1 * hops;
    }
}

} // namespace garnet
} // namespace ruby
} // namespace gem5
