
# Feedback on Project

The project is well-grounded. Gorman et al. (2006) is an appropriate baseline, the replication-then-extension structure is methodologically sound, and the research questions are clear. Some comments

**The S/D/F state transition rules are not specified.** The original Gorman model defines transition probabilities between susceptible, drinking, and former drinker states, but the proposal does not reproduce these rules or explain how the extensions (risk aversion, age susceptibility, social media influencers) modify them. Specifically: does risk aversion lower the probability of transitioning from S to D, raise the probability of D-to-F recovery, or both? Without this, the extensions are listed but not defined.

**The rewiring mechanism conflates two distinct processes.** The proposal describes agents leaving their "immediate social circle when the majority does not agree with their drinking behaviour." This is a _homophily-driven link rewiring_ mechanism — well-studied in adaptive network models of epidemics — which is structurally distinct from the spatial movement rule in the Gorman baseline. The two mechanisms operate on different timescales and produce qualitatively different outcomes: spatial movement changes local density, while network rewiring changes connection topology. You should specify which layer each mechanism operates on, how their timescales compare, and what the rewiring probability is governed by. Also the strategic interaction and game theoretic operationalization is not clear?

**Social media influencers are not mechanistically defined.** "Stronger pressure through social media influencers" is listed as an extension but is never operationalised. Are influencers high-degree hub nodes? Do they broadcast to all agents simultaneously, bypassing the local interaction rule? Do they model peer pressure differently from face-to-face contact? The distinction between local peer interaction and broadcast influence is precisely what makes social media epidemiologically different from spatial contagion, and the mechanism must be specified for the extension to be testable.

**Three-state vs. two-state structure.** The proposal uses Gorman's S/D/F taxonomy, but recent empirical work on alcohol as social contagion using the Framingham Heart Study suggests a three-state SIS-type model (abstainer, moderate drinker, heavy drinker) better fits observed dynamics, because abstainers and heavy drinkers exert _asymmetric_ influence on moderate drinkers. If the model is extended with risk aversion and age susceptibility, distinguishing moderate from heavy drinking becomes meaningful and would strengthen the model's empirical grounding.

**Too many simultaneous extensions for three weeks.** The proposal lists at least five distinct extensions: 2D lattice, network topology, age susceptibility, risk aversion, social media influencers, and rewiring probability. Each of these is a separate structural modification that requires its own validation against the baseline. Running all of them simultaneously risks producing results that cannot be attributed to any single mechanism. Prioritising two or three extensions and conducting the others as robustness checks would produce cleaner, more interpretable findings.

**Suggested Literature**

- Gorman et al. (2006, _Am. J. Public Health_) — the baseline being replicated
- Quiñones et al. (2013, _JASSS_) — binge drinking ABM on social networks with motivational model; the most direct methodological extension of Gorman
- Christakis & Fowler (2024, _Nature Sci. Reports_) — empirical network evidence that alcohol is socially contagious via Framingham cohort; validates the three-state approach
- Lee, Huang & Malik (2025, _arXiv_) — adaptive network rewiring in multi-state drug use propagation with phase transitions; directly relevant to the rewiring extension
- Ball & Britton (2023) — formal analysis of epidemics with preventive rewiring; shows rewiring probability can be a discontinuous control parameter