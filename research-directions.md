# isonome-framework Research Directions

Topics for deep research. Each entry should be investigated thoroughly,
with findings written to /root/isonome-framework/research/<slug>.md.

## Completed Research

### 1. Homeostatic Agent Regulation — 2026-08-14
- **Findings**: research/homeostatic-agent-regulation.md
- **Publishability**: ★★☆ | **Contribution**: ◆◆◇
- **Summary**: Connected mathematical infinitesimal homeostasis (singularity theory) to multi-axis PID control for equilibrium-based agents. Key insights: homeostasis matrix H and det(H)=0 as design constraint, period homeostasis for oscillatory axes, dynamic homeodynamics in multi-timescale systems, gap in multi-agent homeostasis coordination.

### 2. Information-Theoretic Attention — 2026-08-14
- **Findings**: research/information-theoretic-attention.md
- **Publishability**: ★★★ | **Contribution**: ◆◆◇
- **Summary**: Unified rate-distortion theory, context compression (AutoCompressors, semantic compression), surprisal-based attention (energy-gated), and Nixon's semantic rate-distortion for multi-agent communication (quotient POMDP, phase transition at R_crit). Gap: no work connects surprisal-weighted attention with RD-optimal context compression.

### 3. Metacognitive Calibration — 2026-08-14
- **Findings**: research/metacognitive-calibration.md
- **Publishability**: ★★★ | **Contribution**: ◆◆◇
- **Summary**: ECE has Ω(T^{-0.472}) lower bound; CDL achieves O(log T/√T). Temperature scaling insufficient for soft labels. Metacognition ≠ calibration (Oliveira's 5-dim probe, Cacioli's SDT). Self-calibration in open-ended environments needs decision-theoretic metrics (CDL) and metacognitive signals (epistemic vigilance, knowledge boundary).

## Active Research Directions

4. **Multi-Agent Equilibrium**
   - How do multiple equilibrium-based agents coordinate without central control?
   - Can tension axes be shared across agents for fleet-level homeostasis?
   - Literature: Multi-agent reinforcement learning, swarm intelligence

5. **Memory Consolidation Theory**
   - How does biological memory consolidation (systems consolidation) inform artificial memory?
   - What forgetting curve models best predict optimal retention?
   - Literature: Ebbinghaus, McGaugh's memory consolidation, hippocampal replay

6. **Safety-Constrained Autonomy**
   - How do we formally verify that a tension-based controller stays within safety bounds?
   - Can we derive Lyapunov functions for equilibrium-based agent stability?
   - Literature: Control theory, formal verification, AI safety

## Completed Research
<!-- Move entries here once researched -->

## How To Use
The cron job reads this file, picks the next un-researched topic, does deep
research (literature review, mathematical analysis, practical implications),
and writes findings to research/<slug>.md. It then moves the entry to
"Completed Research" and commits.
