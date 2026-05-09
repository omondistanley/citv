# Pathing and Trajectory Implementation Deep Dive (Apr 2026)

This document captures the full summary of our design discussion, the agreed-upon implementation plan, and the success metrics for each phase.

### The Journey So Far: From Rigid Lines to Native Scene Intelligence

Our discussion kicked off when you noticed that generated trajectories were behaving unnaturally in the QA outputs. Paths were defaulting to straight lines, zig-zagging awkwardly between regions, and some trajectories were hanging in mid-air or stubbornly hugging the boundaries of objects rather than moving *through* them. Furthermore, paths would rigidly walk *around* obstacles instead of exhibiting organic, 3D-aware behaviors like jumping or climbing.

As we peeled back the layers of `scene_understanding.py` and the pathing modules, we realized the root cause wasn't a single bug, but a series of architectural traps:
1. **The Region Trap:** We were forcing paths to cross artificial "portals" created by K-Means depth quantization, causing zig-zags.
2. **The 2D Trap:** Our Fast Marching Method (FMM) was operating as a flat 2D water-flow algorithm, completely blind to 3D cliffs and slopes during the search.
3. **The Hardcode Trap:** We were using rigid "obstacle" masks that instantly invalidated any path that touched an object, making "jumping over" or "climbing" mathematically impossible.

To fix this, we pivoted our strategy. We decided to transition the pipeline from **Discrete Geometric Routing** to **Continuous Semantic-Kinematic Routing**.

By leveraging the true 3D surface normals from your depth maps, swapping boundary contours for Medial Axis Skeletons, and using Latent Embeddings to dynamically infer affordances (like "swim" in a "pool"), we designed a system that naturally understands *how* any given actor should traverse *any* given terrain. And, crucially, we designed safeguards (like Elongation Gates and Interaction Pulses) to ensure the math gracefully handles edge cases like perfectly round objects or stationary idle actions without generating visual "blobs".

***

### The Implementation Master Checklist

This is our blueprint. It outlines every architectural upgrade we discussed, why it matters, and exactly what pipeline symptoms it resolves.

#### Phase 1: Topological Integrity & Organic Geometry
*Goal: Eradicate hanging paths, zig-zags, and boundary-hugging.*

*   [ ] **1.1 Connected Component Reachability Graph**
    *   *What it is:* A pre-filter on the `path_walkable` mask that groups navigable pixels into contiguous islands, pruning out micro-puddles.
    *   *What it solves:* **Hanging Paths**. Prevents the FMM solver from starting or ending on a mathematically isolated pixel where it instantly gets stuck.
*   [ ] **1.2 Pole of Inaccessibility & Geodesic Snapping**
    *   *What it is:* Replaces naive Euclidean centroid snapping. We find the deepest interior point of a mask and snap it to the nearest main-navmesh pixel using obstacle-aware geodesic distance.
    *   *What it solves:* Unreachable anchors for U-shaped or hollow objects, ensuring start/end points are physically valid.
*   [ ] **1.3 Medial Axis Transform (Skeletons)**
    *   *What it is:* Replaces OpenCV `findContours` and rigid PCA lines for intra-mask routing. Uses a Distance Transform to extract the perfect 1-pixel "spine" of any shape.
    *   *What it solves:* **Boundary-hugging**. Ensures paths flow down the exact centre of curved hallways, L-shaped desks, and winding regions.
*   [ ] **1.4 The Elongation Gate & Idle Pulses**
    *   *What it is:* Checks the aspect ratio of a mask before extracting a skeleton. If round/stationary, it aborts the path line and emits an `Interaction Pulse` marker.
    *   *What it solves:* **Dots and Blobs**. Prevents the Medial Axis math from collapsing into a single microscopic dot for perfectly round objects (like plates or balls).
*   [ ] **1.5 Continuous FMM Floor Routing (Bypassing Portals)**
    *   *What it is:* Removes the strict requirement to route through artificial "Region Adjacency Graph" borders.
    *   *What it solves:* **Zig-zagging trajectories**. Allows the solver to find the smoothest route across the whole floor.

#### Phase 2: 2.5D Physics & Kinematic Profiling
*Goal: Give the solver true 3D awareness and the ability to jump/climb.*

*   [ ] **2.1 2.5D Traversability Tensor (Surface Normals)**
    *   *What it is:* Calculates the X/Y/Z slope gradient from `metric_depth_m`. Steep slopes dynamically drop the FMM `speed_map` to 0.
    *   *What it solves:* **2D Blindness**. Stops paths from walking off balconies or through walls, forcing them to naturally snake up ramps and avoid physical drops.
*   [ ] **2.2 Soft Semantic Obstacles**
    *   *What it is:* Removes "jumpable" or "climbable" objects from the hard `all_obs` exclusion mask, placing them instead as high-cost spikes in the `cost_map`.
    *   *What it solves:* **Path Blocking**. Allows the solver to calculate that going *over* a low table is sometimes mathematically cheaper than walking 50 pixels around it.
*   [ ] **2.3 Kinematic Signature Extraction**
    *   *What it is:* Analyzes the final `polyline_3d` math. Spikes in Z = `jump`; gradual Z incline = `climb`; zero XY velocity = `idle/hold`.
    *   *What it solves:* **Hardcoded animations**. Dynamically tags segments of the trajectory with the correct physical action based on the geometry of the route.

#### Phase 3: Universal Semantic Affordances
*Goal: Scale to any image (outdoors, fantasy, microscopic) without hardcoding rules.*

*   [ ] **3.1 Latent Space Affordance Mapping**
    *   *What it is:* Passes RAM++ labels (like "lake" or "lava") through your text embedding model and measures Cosine Similarity against Canonical Actions (`walk`, `swim`, `climb`).
    *   *What it solves:* **Rigid label dictionaries**. Natively understands that water allows swimming and lava is high-cost, without requiring `if label == 'water'` statements.
*   [ ] **3.2 Actor-Centric Routing Channels**
    *   *What it is:* Generates different base cost layers (Ground, Fluid, Aerial) depending on the entity type (Person, Boat, Bird).
    *   *What it solves:* Ensures a bird doesn't get constrained by floor geometry and a boat doesn't try to drive on the sidewalk.

#### Phase 4: Cinematic QA Rendering Updates
*Goal: Make the output animations actually reflect the new data.*

*   [ ] **4.1 Dynamic Depth Scaling (Perspective Z-Scale)**
    *   *What it is:* Modifies the rendering loop to scale the spritesheet frame down as Z-depth increases (moving away from the camera).
    *   *What it solves:* Flat, unconvincing 2D sprite movements.
*   [ ] **4.2 Action Timeline Parsing & Speed Matching**
    *   *What it is:* Feeds the Kinematic Signatures (from 2.3) into the renderer to swap animations (walk -> jump -> run) and ties playback FPS to the 3D slope speed.
    *   *What it solves:* Static single-action animations. Characters will now physically slow down and struggle when walking up a steep Z-incline.

### Success Metrics Blueprint

#### Phase 1: Topological Integrity & Organic Geometry
*   **Metric 1.1: Hanging Path Rate (Target: 0%)**
*   **Metric 1.2: Medial Axis Deviation (Target: < 2 pixels)**
*   **Metric 1.3: Degenerate Blob Replacement Rate (Target: 100%)**
*   **Metric 1.4: Artificial Zig-Zag Integral (Target: < 15° turning variance)**

#### Phase 2: 2.5D Physics & Kinematic Profiling
*   **Metric 2.1: Z-Gradient Violations (Target: 0%)**
*   **Metric 2.2: Soft Obstacle Energy Efficiency (Target: 100% Optimal)**
*   **Metric 2.3: Kinematic Signature Tagging (Target: > 95% alignment)**

#### Phase 3: Universal Semantic Affordances
*   **Metric 3.1: Latent Affordance Distance (Target: Cosine similarity > 0.8)**
*   **Metric 3.2: Actor-Constraint Violations (Target: 0%)**

#### Phase 4: Cinematic QA Rendering Updates
*   **Metric 4.1: Perspective Z-Scale Accuracy**
*   **Metric 4.2: Kinematic Speed Synchronization**
*   **Metric 4.3: Timeline Execution Accuracy**