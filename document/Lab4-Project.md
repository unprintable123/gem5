# Lab4: Project
- Two people per group, choose one topic. 


### Topic 1: Topology & routing algorithm
- Implement at least one new topology (e.g. 3D-torus) and the corresponding **deadlock-free** adaptive (non-deterministic) routing algorithm.
- Select **appropriate** baselines and compare performance.

### Topic 2: Flow Control
- Implement at least two new flow control technologies (e.g. escape VC & bubble).
- Select **appropriate** baselines and compare performance.

### Topic 3: Microarchitecture
- Implement "Bypass" and "Multicast"
- Bypass: Add extra links, through which packets can go cross some intermediate nodes. (e.g. shortcut between diagonal nodes)
- Multicast: Send a packet from one node to n nodes by one time rather than n times.
- Select **appropriate** baselines and compare performance.

### Topic 4: Any other project related to the interconnection
- Trace-based traffic pattern
- Circuit-level implementation
- etc.

## Submit
- Source code (only modified files)
  - git patch file.
  - Must include *how to reproduce the results in the report* in a README file.
- A report (PDF)
  - Must include main result comparison (drawn in plots) and analysis
- Presentation (8 min), including:
  - Architecture
  - Implementation
  - Evaluation
  - Division of Labor