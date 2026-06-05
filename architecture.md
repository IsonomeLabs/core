# Isonome Full Architecture

#> Generated from PRD.md + codebase analysis.  
#> PRD version: 0.1 | Code version: 0.2

---

## Diagram 1: High-Level System (PRD View)

```mermaid
flowchart TB
    subgraph INPUT["📥 Input"]
        USER_TASK["Natural Language Task"]
        URDF["Robot URDF"]
        SCENE["Scene JSON (opt)"]
    end

    subgraph CHAMBER1["🏛️ Chamber 1: Deliberative (LLM Orchestrator)"]
        LLM["LLM Orchestrator<br/>isonome/llm/"]
        MANIFEST["Task Manifest (YAML)"]
    end

    subgraph CHAMBER2["🏛️ Chamber 2: Tactical (Morphology Analyzer)"]
        MORPH["Morphology Analyzer"]
        TOPO["Agent Partition<br/>Topology Hash (SHA256)"]
    end

    subgraph CACHE["💾 Calibration Cache"]
        CACHE_KEY["Cache Key:<br/>SHA256(topology + task_type + vla_version)"]
        CACHE_HIT{"Cache Hit?"}
        PKG_CACHED["Cached Policy Package"]
    end

    subgraph CHAMBER3["🏛️ Chamber 3: Operational (Coordination Engine)"]
        FSM["FSM Compiler<br/>Guards / Events / Merge Strategy"]
        MERGE["Action Merger<br/>Priority | Weighted Avg | Nullspace"]
    end

    subgraph SIM["🔬 Data Plane: Simulation Software"]
        ISAAC["Isaac Lab Environments<br/>(per-agent isolated)"]
        MUJOCO["MuJoCo MJX Fallback<br/>(contact-rich tasks)"]
        CALIB["Parallel Calibration<br/>CMA-ES / Differentiable Sim"]
        COMP["Composition Validation<br/>1000 episodes"]
        AUTO["Auto-Adjustment Engine<br/>(max 5 iterations)"]
    end

    subgraph CHAMBER4["🏛️ Chamber 4: Reactive (Reflex Layer)"]
        REFLEX["Reflex Layer<br/>1 kHz | Dedicated CPU Thread"]
        JOINT_LIMITS["Joint Limit Clamping"]
        VELOCITY["Velocity Limits"]
        EMERGENCY["Emergency Stance / E-Stop"]
    end

    subgraph RUNTIME["⚡ VLA Runtime (Edge Execution)"]
        VLA_LOAD["VLA Loader<br/>pi0 | OpenVLA | SmolVLA"]
        INF_CTX["N Inference Contexts<br/>(one per agent)"]
        OBS_BUF["Observation Ring Buffers<br/>(size 10)"]
        SHARED["Shared World State<br/>(lock-free dict @ 200 Hz)"]
    end

    subgraph OUTPUT["📤 Output"]
        POLICY_PKG["Certified Policy Package (.zip)"]
        ROS2["ROS2 Nodes / Standalone"]
        HARDWARE["Hardware Commands"]
    end

    USER_TASK --> LLM
    URDF --> LLM
    SCENE --> LLM
    LLM --> MANIFEST

    URDF --> MORPH
    MANIFEST --> MORPH
    MORPH --> TOPO

    TOPO --> CACHE_KEY
    CACHE_KEY --> CACHE_HIT
    CACHE_HIT --"Yes"--> PKG_CACHED
    CACHE_HIT --"No"--> ISAAC

    MANIFEST --> FSM
    TOPO --> FSM

    ISAAC --> CALIB
    CALIB --> COMP
    COMP --"< 99% success"--> AUTO
    AUTO --> ISAAC
    COMP --"Contact failures > 10%"--> MUJOCO
    MUJOCO --> COMP
    COMP --"> 99% success"--> POLICY_PKG
    PKG_CACHED --> POLICY_PKG

    FSM --> MERGE
    VLA_LOAD --> INF_CTX
    INF_CTX --> OBS_BUF
    OBS_BUF --> MERGE
    SHARED --> MERGE

    MERGE --> REFLEX
    REFLEX --> JOINT_LIMITS
    JOINT_LIMITS --> VELOCITY
    VELOCITY --> EMERGENCY
    EMERGENCY --> ROS2
    EMERGENCY --> HARDWARE

    POLICY_PKG --> VLA_LOAD
    POLICY_PKG --> FSM
    POLICY_PKG --> REFLEX
```

---

## Diagram 2: Agent Layer Pipeline (Code View)

```mermaid
flowchart LR
    subgraph AGENT["🤖 Agent (isonome.core.agent.Agent)"]
        direction TB

        subgraph BOOT["Boot Sequence"]
            B1["soma.boot()"] --> B2["jepa.boot()"]
            B2 --> B3["cortex.boot()"]
            B3 --> B4["reflex.boot()"]
            B4 --> B5["plasticity.boot()"]
        end

        subgraph TICK["Tick Loop (~100 Hz via reflex.tick_period)"]
            direction LR
            T1["1. Perceive<br/>soma.perceive()"] --> T2["2. Advise<br/>cortex.advise()"]
            T2 --> T3["3. Build Prompt<br/>cortex.build_prompt()"]
            T3 --> T4["4. Deliberate<br/>jepa.deliberate()"]
            T4 --> T5["5. Apply Kernel<br/>soma.apply_kernel()"]
            T5 --> T6["6. Process<br/>reflex.process()"]
            T6 --> T7["7. Act<br/>soma.act()"]
            T7 --> T8["8. Observe<br/>soma.observe_result()"]
            T8 --> T9["9. Buffer<br/>cortex.buffer.add()"]
        end

        subgraph SHUTDOWN["Shutdown Sequence"]
            S1["plasticity.shutdown()"] --> S2["cortex.shutdown()"]
            S2 --> S3["jepa.shutdown()"]
            S3 --> S4["reflex.shutdown()"]
            S4 --> S5["soma.shutdown()"]
        end
    end

    subgraph SAFETY["🛡️ Safety Governor (isonome.core.safety)"]
        SG["SafetyGovernor"]
        MODE["AgentMode<br/>BOOT | IDLE | RUNTIME | CALIBRATING | SAFE_STOP"]
        ESTOP["EmergencyStop Exception"]
    end

    TICK -."can_execute?".-> SG
    SG -."raises".-> ESTOP
    ESTOP -."caught by".-> AGENT
```

---

## Diagram 3: Layer Detail — Data Transformations

```mermaid
flowchart TB
    subgraph LAYER_S["🦴 SomaLayer (isonome.core.layers.soma)"]
        URDF_FILE["URDF File"]
        NAIVE["NaiveMapper<br/>canonical 14-DOF → robot N-DOF"]
        KERNEL["SomaKernel (nn.Module)<br/>residual correction network"]
        RAW["RawSensorState<br/>proprioception + camera_frames"]
    end

    subgraph LAYER_J["🧠 JEPALayer (isonome.core.layers.jepa)"]
        VLA_PI0["pi0 / pi0-fast<br/>(lerobot)"]
        VLA_OPENVLA["OpenVLA<br/>(transformers)"]
        VLA_SMOLVLA["SmolVLA<br/>(transformers)"]
        CANONICAL["CanonicalActionChunk<br/>[chunk_size, 14]"]
    end

    subgraph LAYER_C["🧩 CortexLayer (isonome.core.layers.cortex)"]
        DISC_BUF["DiscrepancyBuffer<br/>Ring buffer (size 10)"]
        ADVICE["CortexAdvice[]<br/>natural language corrections"]
        PROMPT["Prompt String<br/>task + advice injection"]
    end

    subgraph LAYER_R["⚡ ReflexLayer (isonome.core.layers.reflex)"]
        INTERP["ActionInterpolator<br/>policy_freq → control_freq"]
        ENFORCER["SafetyEnforcer<br/>clamp to JointLimits"]
        SAFE["SafeMotorCommand[]<br/>was_clamped + emergency_stop"]
    end

    subgraph LAYER_P["🔧 PlasticityLayer (isonome.core.layers.plasticity)"]
        KERN_DIR["~/.isonome/kernels/"]
        META["KernelMetadata<br/>version | episodes | robot_hash"]
        LOAD_PT["load_kernel(.pt)"]
        SAVE_PT["save_runtime_state(.pt)"]
    end

    URDF_FILE --> NAIVE
    NAIVE --> KERNEL
    KERNEL -->|"corrected commands"| LAYER_R
    RAW -->|"RAW only"| LAYER_J
    RAW -->|"perceive()"| LAYER_S

    VLA_PI0 --> CANONICAL
    VLA_OPENVLA --> CANONICAL
    VLA_SMOLVLA --> CANONICAL
    CANONICAL -->|"input to kernel"| KERNEL

    ADVICE -->|"injected into"| PROMPT
    PROMPT -->|"deliberate(raw, prompt, advice)"| LAYER_J
    DISC_BUF -->|"advise() reads"| ADVICE

    INTERP --> ENFORCER
    ENFORCER --> SAFE
    SAFE -->|"executed by soma.act()"| LAYER_S

    KERN_DIR --> LOAD_PT
    LOAD_PT -->|"loads into"| KERNEL
    KERNEL -->|"persists via"| SAVE_PT
    SAVE_PT --> KERN_DIR
    META -->|"attached to"| SAVE_PT
```

---

## Diagram 4: Module Dependency Map

```mermaid
flowchart TB
    subgraph PKG["📦 isonome package"]
        direction TB

        subgraph CORE["🔧 isonome.core"]
            AGENT_M["agent.py<br/>Agent class — layer orchestrator"]
            APP_M["app.py<br/>IsonomeApp — lifecycle + signals"]
            CONFIG_M["config.py<br/>AppConfig, SimConfig"]
            STATE_M["state.py<br/>Pydantic state models (v0.2 + v0.1 legacy)"]
            SAFETY_M["safety.py<br/>SafetyGovernor, AgentMode, EmergencyStop"]
            BUS_M["bus.py<br/>Event bus"]

            subgraph LAYERS["layers/"]
                BASE_L["base.py<br/>LayerBase (ABC)"]
                SOMA_L["soma.py<br/>SomaLayer, SomaKernel, NaiveMapper"]
                JEPA_L["jepa.py<br/>JEPALayer, VLABackend, load_vla()"]
                CORTEX_L["cortex.py<br/>CortexLayer, DiscrepancyBuffer"]
                REFLEX_L["reflex.py<br/>ReflexLayer, ActionInterpolator, SafetyEnforcer"]
                PLAST_L["plasticity.py<br/>PlasticityLayer, KernelMetadata"]
            end
        end

        subgraph BRIDGE["🌉 isonome.bridge"]
            SIM_B["sim.py<br/>SimBridge (PyBullet)"]
            HW_B["hardware.py<br/>HardwareBridge (ABC), StubHardwareBridge"]
        end

        subgraph LLM["🗣️ isonome.llm"]
            LLM_CLIENT["client.py<br/>LLM client"]
            LLM_SWARM["swarm.py<br/>LLM swarm orchestration"]
            LLM_CACHE["cache.py<br/>Calibration cache"]
        end

        subgraph SAFETY_PKG["🛡️ isonome.safety"]
            GOV["governor.py"]
            SANDBOX["sandbox.py"]
            WATCHDOG["reflex_watchdog.py"]
        end

        subgraph PRESETS["📋 isonome.presets"]
            PRESET_BASE["base.py<br/>PresetBase"]
            PRESET_LOADER["loader.py"]
            BUILT_IN["built_in/<br/>pet.py | patrol.py"]
        end

        subgraph UTILS["🛠️ isonome.utils"]
            LOGGING["logging.py<br/>structured logging"]
        end

        subgraph CLI["⌨️ isonome.cli"]
            CLI_M["cli.py<br/>Typer CLI: init | sim | run | deploy"]
        end
    end

    APP_M --> AGENT_M
    AGENT_M --> SOMA_L
    AGENT_M --> JEPA_L
    AGENT_M --> CORTEX_L
    AGENT_M --> REFLEX_L
    AGENT_M --> PLAST_L
    AGENT_M --> SAFETY_M

    SOMA_L --> BASE_L
    JEPA_L --> BASE_L
    CORTEX_L --> BASE_L
    REFLEX_L --> BASE_L
    PLAST_L --> BASE_L

    SOMA_L --> STATE_M
    JEPA_L --> STATE_M
    CORTEX_L --> STATE_M
    REFLEX_L --> STATE_M
    PLAST_L --> STATE_M

    SOMA_L -->|"loads URDF"| SIM_B
    SOMA_L -->|"or hardware"| HW_B

    JEPA_L -->|"loads policy"| LLM_CLIENT
    PLAST_L -->|"kernel cache"| LLM_CACHE

    SAFETY_M --> GOV
    REFLEX_L -->|"emergency stop"| WATCHDOG

    APP_M --> CONFIG_M
    AGENT_M --> CONFIG_M
    CORE --> LOGGING
    CLI_M --> APP_M
    PRESET_LOADER --> PRESET_BASE
    BUILT_IN --> PRESET_BASE
```

---

## Diagram 5: State Machine — Agent Modes

```mermaid
stateDiagram-v2
    [*] --> BOOT : Agent()
    BOOT --> IDLE : boot() completes
    IDLE --> RUNTIME : run() starts
    IDLE --> CALIBRATING : load_kernel()
    CALIBRATING --> RUNTIME : kernel loaded
    RUNTIME --> SAFE_STOP : EmergencyStop raised
    RUNTIME --> IDLE : shutdown()
    SAFE_STOP --> IDLE : reset + shutdown()
    IDLE --> [*] : process exit
```

---

## Diagram 6: Tick Invariant (One-Frame Delay)

```mermaid
sequenceDiagram
    autonumber
    participant S as SomaLayer
    participant C as CortexLayer
    participant J as JEPALayer
    participant R as ReflexLayer
    participant B as Bridge (Sim/HW)

    Note over S,B: Invariant: JEPA never sees corrected or post-execution state

    rect rgb(230,245,255)
        Note over S: Step 1: Read RAW
        S->>S: perceive() → RawSensorState
    end

    rect rgb(255,245,230)
        Note over C: Step 2: Build advice
        C->>C: advise() → CortexAdvice[]
        C->>C: build_prompt(raw, advice) → prompt
    end

    rect rgb(230,255,230)
        Note over J: Step 3: Deliberate from RAW
        J->>J: deliberate(raw, prompt, advice) → CanonicalActionChunk
    end

    rect rgb(255,230,245)
        Note over S: Step 4: Correct for THIS body
        S->>S: apply_kernel(canonical, raw) → CorrectedMotorCommand
        alt No kernel loaded
            S->>S: naive_map(canonical) → fallback
        end
    end

    rect rgb(255,255,230)
        Note over R: Step 5: Interpolate + Enforce safety
        R->>R: process(corrected) → SafeMotorCommand[]
    end

    rect rgb(240,240,240)
        Note over S,B: Step 6: Execute
        S->>B: act(safe_commands)
        B-->>S: physics step / motor response
    end

    rect rgb(230,245,255)
        Note over S: Step 7: Observe result for NEXT tick
        S->>S: observe_result() → ExecutionResult
        S->>C: cortex.buffer.add(intended, result, raw)
    end
```

---

## Diagram 7: Simulation / Calibration Pipeline (PRD Future State)

```mermaid
flowchart TB
    subgraph INPUT["Input"]
        M["Task Manifest YAML"]
        U["Robot URDF"]
        VLA["VLA Weights (.pt)"]
    end

    subgraph STEP1["Step 1: Parallel Isolated Calibration"]
        STRIP["URDF Stripper<br/>per-agent joint subsets"]
        ENV["Isaac Lab Env Spawner<br/>256 parallel envs"]
        DR["Domain Randomization<br/>mass | friction | damping | lighting"]
        CMA["CMA-ES / Differentiable Sim<br/>optimize W·a_vla + b"]
        ISO_PASS{"> 95% isolated<br/>success rate?"
        }
    end

    subgraph STEP2["Step 2: Composition Validation"]
        FULL["Full-Body Sim<br/>all agents + coordinator + reflex"]
        EP["1000 Randomized Episodes"]
        LOG["Failure Mode Logger<br/>agent | guard | merge | reflex | physics"]
        COMP_PASS{"> 99% composition<br/>success rate?"
        }
    end

    subgraph STEP3["Step 3: Auto-Adjustment"]
        ANALYZE["Pattern Analysis"]
        ADJUST["Adjust:<br/>guard thresholds | merge strategy | reflex gains | prompt"]
        MAX_ITER{"Iteration < 5?"}
    end

    subgraph STEP4["Step 4: Export"]
        CERT["Certified Policy Package (.zip)"]
        ARTIFACTS["Contains:<br/>manifest | per-agent policies | coordinator config |<br/>reflex gains | sim metrics | certification video | launcher"]
    end

    M --> STRIP
    U --> STRIP
    VLA --> CMA
    STRIP --> ENV
    ENV --> DR
    DR --> CMA
    CMA --> ISO_PASS
    ISO_PASS --"No"--> CMA
    ISO_PASS --"Yes"--> FULL

    FULL --> EP
    EP --> LOG
    LOG --> COMP_PASS
    COMP_PASS --"Yes"--> CERT
    COMP_PASS --"No"--> ANALYZE
    ANALYZE --> ADJUST
    ADJUST --> MAX_ITER
    MAX_ITER --"Yes"--> FULL
    MAX_ITER --"No"--> CERT
    CERT --> ARTIFACTS
```

---

## Diagram 8: Runtime ROS2 Topic Topology

```mermaid
flowchart LR
    subgraph AGENTS["Per-Agent VLA Instances"]
        A1["agent/locomotion"]
        A2["agent/right_arm"]
        A3["agent/gripper"]
    end

    subgraph TOPICS["ROS2 Topics"]
        T1["/isonome/agents/{id}/partial_action<br/>(debug)"]
        T2["/isonome/coordinator/full_action<br/>(pre-reflex)"]
        T3["/isonome/reflex/safe_action<br/>(final hardware command)"]
        T4["/isonome/fsm/phase<br/>(current state)"]
    end

    subgraph COORD["Coordination Engine"]
        FSM["FSM Executor<br/>200 Hz"]
        MERGE["Action Merger"]
    end

    subgraph REFLEX["Reflex Layer"]
        R1["Joint Limit Clamp"]
        R2["Velocity Enforcer"]
        R3["Emergency Stance"]
    end

    A1 --> T1
    A2 --> T1
    A3 --> T1
    T1 --> MERGE
    MERGE --> T2
    T2 --> REFLEX
    REFLEX --> T3
    FSM --> T4
    T4 --> A1
    T4 --> A2
    T4 --> A3
```

---

## Diagram 9: Topology Vector (32-Dimensional)

```mermaid
flowchart LR
    URDF["URDF Parse"] --> VEC["32-D Topology Vector"]

    subgraph DIMS["Dimensions"]
        D0["0-2: Base Type<br/>fixed / diff-drive / holonomic / legged"]
        D1["3-5: Max DOF/arm<br/>/20"]
        D2["6-8: Max DOF/leg<br/>/20"]
        D3["9-11: End-Effector Type<br/>one-hot"]
        D4["12-14: Link Length Ratios (arm)<br/>log scale"]
        D5["15-17: Mass Ratios (arm vs torso)<br/>/10"]
        D6["18-20: Max Torque Ratios<br/>/100"]
        D7["21-23: Workspace Volume<br/>log scale"]
        D8["24-26: Joint Damping Mean<br/>/1.0"]
        D9["27-29: Friction Coeff Mean<br/>/1.0"]
        D10["30: Head/Gaze Joints<br/>binary"]
        D11["31: Force Sensors<br/>binary"]
    end

    VEC --> D0
    VEC --> D1
    VEC --> D2
    VEC --> D3
    VEC --> D4
    VEC --> D5
    VEC --> D6
    VEC --> D7
    VEC --> D8
    VEC --> D9
    VEC --> D10
    VEC --> D11

    D0 --> HASH["SHA256 Topology Hash"]
    D1 --> HASH
    D2 --> HASH
    D3 --> HASH
    D4 --> HASH
    D5 --> HASH
    D6 --> HASH
    D7 --> HASH
    D8 --> HASH
    D9 --> HASH
    D10 --> HASH
    D11 --> HASH
```
