# Literature notes and search log (idea #1: yard-sale model)

Compiled 2026-09-30 during the feasibility investigation. Every claim of "already published" or "not found" below
is backed by the searches listed at the end.

## A. Exact references

**Origins and canonical kinetic-exchange / multiplicative models**

1. B. Hayes, "Follow the money", *American Scientist* **90**(5), 400 (2002) — popular origin of the yard-sale game.
2. S. Ispolatov, P. L. Krapivsky, S. Redner, "Wealth distributions in asset exchange models", *Eur. Phys. J. B* **2**, 267–276 (1998).
3. A. Chakraborti, "Distributions of money in model markets of economy", *Int. J. Mod. Phys. C* **13**(10), 1315–1321 (2002) — the yard-sale model (YSM) as now used.
4. J.-P. Bouchaud, M. Mézard, "Wealth condensation in a simple model of economy", *Physica A* **282**, 536–545 (2000); arXiv:cond-mat/0002374.
5. A. Dragulescu, V. M. Yakovenko, "Statistical mechanics of money", *Eur. Phys. J. B* **17**, 723 (2000); "Exponential and power-law probability distributions of wealth and income in the United Kingdom and the United States", *Physica A* **299**, 213 (2001). arXiv:cond-mat/0001432, cond-mat/0103544.
6. A. Chatterjee, B. K. Chakrabarti, "Kinetic exchange models for income and wealth distributions", *Eur. Phys. J. B* **60**, 135–149 (2007).
7. M. Greenberg, H. O. Gao, "Twenty-five years of random asset exchange modeling", *Eur. Phys. J. B* **97**, 78 (2024).

**Condensation, WAA and phase transitions**

8. C. F. Moukarzel, S. Gonçalves, J. R. Iglesias, M. Rodríguez-Achach, R. Huerta-Quintanilla, "Wealth condensation in a multiplicative random asset exchange model", *Eur. Phys. J. Spec. Top.* **143**, 75–79 (2007) — fixed bias, first-order transition to absolute oligarchy.
9. C. Chorro, "A simple probabilistic approach of the yard-sale model", *Stat. Probab. Lett.* **112**, 35–40 (2016).
10. C. Börgers, C. Greengard, "A new probabilistic analysis of the yard-sale model", arXiv:2308.01485 (2023); "Local wealth condensation for yard-sale models with wealth-dependent biases", arXiv:2406.10978 (2024).
11. B. M. Boghosian, A. Devitt-Lee, M. Johnson, J. Li, J. A. Marcq, H. Wang, "Oligarchy as a phase transition: the effect of wealth-attained advantage in a Fokker–Planck description of asset exchange", *Physica A* **476**, 15–37 (2017); arXiv:1511.00770. Critical condensed fraction `c∞ = 1 − τ∞/ζ`.
12. B. M. Boghosian, A. Devitt-Lee, H. Wang, "The growth of oligarchy in a yard-sale model of asset exchange: a logistic equation for wealth condensation", arXiv:1608.05851 (2016); Proc. 1st Int. Conf. Complex Information Systems, doi:10.5220/0005956501870193.
13. D. W. Cohen, B. M. Boghosian, "Bounding the approach to oligarchy in a variant of the yard-sale model", arXiv:2310.16098 (2023, rev. 2024).
14. C. F. Moukarzel, "Multiplicative asset exchange with arbitrary return distributions", arXiv:1108.0386, *J. Stat. Mech.* (2011) — condensation criterion `⟨ln η⟩ < 0`.
15. K. Kobayashi, K. Takaoka, "Some asymptotic results for the yard-sale model of asset exchange", Hitotsubashi University CFR Working Paper G-1-17 (2016) — unfair games, heterogeneous transaction rates/intensities, proportional capital tax.

**Networks, dynamics, heterogeneity**

16. R. Bustos-Guajardo, C. F. Moukarzel, "Yard-sale exchange on networks: wealth sharing and wealth appropriation", *J. Stat. Mech.* (2012) P12009; arXiv:1208.4409 — 1-D rings, 2-D lattices, random graphs with variable coordination; stable-phase critical interface topology-independent; unstable-phase condensation *extensive*, network-dependent.
17. G. L. Kohlrausch, S. Gonçalves, "Wealth distribution on a dynamic complex network", *Physica A* **632**, 129339 (2024); arXiv:2302.03677 — wealth-dependent rewiring; condensation of wealth *and* connections; social-protection factor.
18. G. L. Kohlrausch, S. Gonçalves, "Does a rising tide lift all boats? A wealth exchange model on a dynamic network with economic growth", arXiv:2607.25874 (2026) — dynamic network + growth.
19. L. Giordano, I. Cortés, S. Gonçalves, M. F. Laguna, "Limiting risk to reduce inequality: insights from the yard-sale model", *Physica A* **676** (2025); arXiv:2508.06650 — heterogeneous risk caps.
20. G. Villafañe, L. Giordano, M. F. Laguna, "Wealth inequality in agent-based economies: the dominant role of social protection over growth", arXiv:2508.06666 (2025).

**Taxes / redistribution**

21. R. Bustos-Guajardo, C. F. Moukarzel, "Wealth distribution under yard-sale exchange with proportional taxes", *Int. J. Mod. Phys. C* **27**(08), 1650094 (2016) — wealth-proportional tax; asymptotically Gaussian tails, restricted-range power law with exponent 3/2.
22. H. Lima, A. R. Vieira, C. Anteneodo, "Nonlinear redistribution of wealth from a stochastic approach", *Chaos Solitons Fractals* **163**, 112485 (2022).
23. I. N. Barros, M. L. Martins, "Effects of taxes, redistribution actions and fiscal evasion on wealth inequality: an agent-based model approach", *Physica A* **679**, 130960 (2025); arXiv:2501.08573.
24. B. Boghosian, C. Börgers, "The mathematics of poverty, inequality, and oligarchy", *SIAM News* **56**(8) (2023).
25. (Unverified authors, listed for completeness of the search) "A Fokker–Planck approach to a stochastic multiplicative wealth model with taxation and redistribution", arXiv:2607.11755 (2026).

## B. What is already established (mapped to the team's proposed extensions)

| Proposed extension | Published status | Reference |
|---|---|---|
| Fair YSM, equal starts, fraction-of-poorer transfer | Gini → 1 almost surely; martingale proof | 9, 10 |
| Fixed bias p ≠ 0.5 | First-order transition to absolute oligarchy | 8 |
| Wealth-dependent bias (WAA) | Second-order transition; `c∞ = 1 − τ∞/ζ`; logistic growth in time | 11, 12 |
| Gini as Lyapunov functional / approach bounds | Proved / bounded | 11, 13 |
| Wealth tax + equal redistribution | Stationary distribution; Gaussian tails, 3/2 power law at weak tax; policy studies | 11, 21, 22, 23 |
| Nonlinear / income-based redistribution, evasion | Studied | 22, 23 |
| Transaction size / risk range | Risk-capped heterogeneity | 14, 19 |
| Static topology (ring, lattice, random graph, ⟨k⟩ scan) | Critical interface topology-independent; extensive condensation | 16 |
| Scale-free/small-world/BA networks | BA hubs accelerate single-agent concentration (prototype result); general network YSM literature | 16 (+ prototype) |
| Dynamic network (rewiring) | Wealth-dependent rewiring; wealth + connection condensation; +growth variant | 17, 18 |
| Initial conditions (equal/random/unequal) | Standard sensitivity checks | 19, 21 |

## C. Gap analysis (what the prototype targets as *novel*)

1. **Tax base: trade gains vs wealth.** No located paper taxes *gains* of individual trades in a kinetic
   asset-exchange model. All tax studies use a wealth/income base (21–25). The contrast is analytically clean
   (`E[Δln w] = ½ln(1−f) + ½ln(1+f(1−s)) < 0` for every gain-tax rate s < 1) and numerically dramatic (prototype:
   τ = 0.1 wealth tax → Gini 0.19 stationary; τ = 0.1 gain tax → Gini 0.94 and rising, 360× less revenue at the end
   of a 12 000-sweep run).
2. **Random, wealth-independent rewiring as a partner-access probe.** Published dynamic-network models rewire by
   wealth (17, 18). Keeping the degree distribution fixed and varying only partner turnover isolates a mechanism:
   prototype shows max degree unchanged (6) while the richest agent's distinct partners go 3.75 → 330 and its share
   0.020 → 0.229. This is a different and sharper question than "does rewiring matter".
3. **Condensation-time scaling `t_cond(N)`.** Published analytics bound the Gini approach (13) or describe
   decorrelation times in the stable phase (16); a systematic report of the first-passage time to a fixed wealth
   share in the condensing phase appears absent from the reviewed set. Prototype: `t_cond ≈ 9.8·N^1.01` (p = 0.5,
   f = 0.3) and `≈ 8.1·N^0.97` (p = 0.55).

**Residual novelty risk.** Searches for "transaction tax"/"Tobin tax" in econophysics wealth models returned only
finance-policy literature, not kinetic-exchange studies, but a paper could exist under other terminology. A final
targeted search (Google Scholar, Scopus if available) is recommended before the claim is locked.

## D. Lecture-overlap evidence (UWA CITS4403)

* Public repo of a past cohort, `github.com/LDY681/CITS4403-labs`, lists: Boids, graph labs (weeks 2–3), cellular
  automata (weeks 4–5), agent-based modelling (weeks 7–8), evolution (weeks 10–11), self-organised criticality
  (week 12). **No wealth-distribution or yard-sale lab found.**
* The unit's forbidden list includes *Sugarscape model of wealth distribution*, which indicates wealth-distribution
  modelling **is** taught (typically in the ABM block). Most banned topics map onto classic lab models
  (Game of Life, Schelling, Boids, evolution, traffic jams, Prisoner's Dilemma).
* UWA handbook listings (2026 web search) confirm CITS4403 "Computational Modelling" exists as a 6-point unit but
  do not expose lecture content.
* **Assessment:** the *mechanism* (kinetic exchange) is not banned or (on available evidence) labbed; the
  *phenomenon* (emergent wealth inequality in an ABM) is adjacent to taught material. The team must explicitly
  differentiate from Sugarscape and foreground its novel claims. This is the main originality risk for idea #1.

## E. Search log (queries actually run, 2026-09-30)

1. `Boghosian oligarchy phase transition yard-sale model wealth condensation` → arXiv:1511.00770, 1608.05851, 2310.16098, Physica A 476 15–37.
2. `yard-sale model network topology wealth condensation complex networks study` → arXiv:1208.4409 (networks), 2302.03677 (dynamic network), 2607.25874 (dynamic network + growth), 0901.2857 (directed networks; authors not verified, not cited).
3. `Moukarzel 2007 wealth condensation first order phase transition fixed bias asset exchange citation` → ref 8 details, ref 14, ref 16.
4. `UWA CITS4403 complex systems unit outline wealth distribution kinetic exchange teaching materials` → handbook pages, LDY681 CITS4403-labs repo.
5. `Bustos-Guajardo Moukarzel "proportional taxes" yard-sale wealth distribution critical tax rate` → ref 21 abstract; Barros & Martins 2025.
6. `"Limiting risk to reduce inequality" Yard-Sale model Physica A 2025 abstract transaction size` → ref 19 (arXiv:2508.06650).
7. `transaction tax Tobin tax kinetic exchange model wealth distribution condensation prevention` → finance-policy results only; no kinetic-exchange transaction-tax paper found.
8. `"kinetic exchange" OR "yard-sale" model "tax" on gains versus wealth redistribution Barros Martins fiscal evasion 2025` → refs 22, 23; arXiv:2607.11755.
9. `Cohen Boghosian "Bounding the approach to oligarchy" yard-sale exponential bound N agents` → ref 13 details; SIAM News overview (ref 24).
10. `CITS4403 Computational Modelling UWA github lab notebooks agent based models units` → lab-topic evidence in §D.
11. Fetch `https://arxiv.org/abs/1208.4409` (Bustos-Guajardo & Moukarzel 2012 full abstract), `https://arxiv.org/abs/2607.25874` (authors: Kohlrausch & Gonçalves), `https://github.com/LDY681/CITS4403-labs` (file list).