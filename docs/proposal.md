# Project Proposal (submitted 5 June 2026)

## Abstract

Drinking behaviour emerges from a combination of individual decision making,
peer influence and the availability of drinking environments. We will replicate
and extend the agent-based model by Gorman et al. (2006), in which agents move
on a one-dimensional lattice, interact locally, and adjust their drinking
status. In the second iteration of the model in the paper, a bar is also
introduced to attract drinkers and alter diffusion. In the original paper,
agents can take three statuses: susceptible nondrinkers (S), current drinkers
(D), and former drinkers (D), and change their drinking status with a given
probability. In our research, we aim to replicate the key results from the
original paper, and extend the model both with new architecture and additional
features and attributes of the agents. Our research questions are: how do
bounded rationality, risk/loss aversion, and alcohol-outlet attraction affect
the emergence, clustering, and persistence of drinking behaviour? After
reproducing the baseline experiments on movement, relapse/resistance and bar
attraction, we extend the model in different ways: firstly, we will study how
the same interactions can be modeled using a two-dimensional lattice and/or a
network model. Secondly, we aim to study how adding attributes such as
age-determined susceptibility, risk aversion, and stronger pressure through
social media influencers will impact the propagation of the drinking behavior
through the network. Lastly, we will calibrate agents with a probability of
leaving their immediate social circle when the majority of it does not agree
with the agent's drinking behaviors, reflecting rewiring probability. We will
measure drinking prevalence, relapse, susceptible survival, spatial clustering,
and sensitivity of outcomes to behavioral and environmental parameters. Lastly,
we will perform sensitivity analysis where we will study how varying values of
multiple model parameters, such as number of agent's neighbors or the threshold
for rewiring will affect the results and drinking contagion. This way, we will
identify which behavioral and environmental parameters most strongly determine
these emergent outcomes. With this research, we aim to provide more
understanding into how drinking behaviors emerge and spread among individuals,
and how different factors and characteristics play a role in the process.

## Why Agent-Based Modelling?

Drinking behavior is not something that occurs in isolation. Rather, it
emerges from individuals making decisions, interacting with their friends,
and responding to environmental stimuli. Because the objective of this
project is to study how these individual-level interactions generate
population-level patterns, Agent-Based Modeling (ABM) is the most suitable
approach. ABMs represent each individual as a separate agent with their own
attributes, behavioral rules and social connections. This allows agents to
differ from one another in meaningful ways rather than having an entire
population be identical, which allows to further study properties emergent
from local differences. In our project, we want agents to vary in
characteristics such as age-based susceptibility, risk aversion, social
influence, and attraction to alcohol outlets, which significantly affect the
patterns in drinking behavior.

The alternative modeling approaches are less suitable for addressing our
research questions. Ordinary Differential Equations (ODEs) models describe
transition between drinking states, but their drawback is that they assume a
well-mixed population where every individual has an equal probability of
interacting with every other individual. Similarly, Markov Chain models can
capture probabilistic transition between states, but they do not explicitly
represent social networks, spatial environments, or different agent
characteristics. The original model by Gorman et al. (2006) does demonstrate
the importance of these factors. Drinking behavior spreads through local
contact between neighboring agents, while environmental features such as bars
influence movements patterns and exposure, with different agents having
different levels of influence and pressure put on them depending on their
surroundings. These processes depend on individual properties and
interactions between different agents, which makes it difficult to represent
with other models. In addition, ABMs also make is easier to simulate and test
various scenarios, as well as parameter sensitivity to see how the behaviors
vary while the parameter values change.
