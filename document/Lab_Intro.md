# AI+X Lab: NoC Implementation and Design with Gem5 and Garnet

## Introduction

In this lab, you will try to implement existing NoC systems, including different topology, routing algorithm, flow-control mechanism, etc. You will even have the opportunity of designing your own NoC system in the end of the lab.

The lab uses *Gem5* and *Garnet* as the basic platform. *Gem5* is a open-source simulator for both system-level architecture and processor microarchitecture, and *Garnet* is an on-chip network model for diverse interconnect systems that builds upon Gem5.

The lab will contain 5 labs: lab 0 to lab 4, which is listed below:

## Labs

- Lab 0 (preliminary)
  - Install and configure the environment, including Gem5 and Garnet
- Lab 1
  - Understand the basic mechanism and usage of Garnet.
- Lab 2
  - Learn how to analysis network performance.
- Lab 3
  - Implement existing topology and flow-control.
- Lab 4 (project)
  - Design/Implement your own topology/flow-control/arch/etc.

## Tips

- Do not forget to *rm build* before re-building during installation of gem5 (if you encounter any error/problem).

- scons has *incremental compilation*, so once you successfully builds gem5, re-building will be much faster after you modify some codes.

- The reports submitted for the labs needs to have experiment results / analysis as indicated in each lab.

- The following files may be related to the labs:
  - src/mem/ruby/network/garnet/*
  - configs/example/garnet_synth_traffic.py
  - configs/network/Network.py
  - configs/topologies/*

- Feel free to ask in the Wechat group if you have any question.

- Enjoy and have fun!
