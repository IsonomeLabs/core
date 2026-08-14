# isonome-framework Research Directions

Topics for deep research. Each entry should be investigated thoroughly,
with findings written to /root/isonome-framework/research/<slug>.md.

## Completed Research

### 1. Homeostatic Agent Regulation — 2026-08-14
- **Findings**: research/homeostatic-agent-regulation.md
- **Publishability**: ★★☆ | **Contribution**: ◆◆◇
- **Summary**: Connected mathematical infinitesimal homeostasis (singularity theory) to multi-axis PID control for equilibrium-based agents. Key insights: homeostasis matrix H and det(H)=0 as design constraint, period homeostasis for oscillatory axes, dynamic homeodynamics in multi-timescale systems, gap in multi-agent homeostasis coordination.

## Active Research Directions

2. **Information-Theoretic Attention**
   - What is the optimal compression ratio for context windows under different task types?
   - How does surprisal-based attention compare to learned attention (transformer-style)?
   - Literature: Shannon's information theory, attention mechanisms in transformers

3. **Metacognitive Calibration**
   - What are the theoretical limits of self-calibration in learned systems?
   - Can ECE ever converge to zero in open-ended environments?
   - Literature: Guo et al. "On Calibration of Modern Neural Networks", Bayesian deep learning

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
