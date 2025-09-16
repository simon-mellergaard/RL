# Exercises

## 6.6.1 Exercise - Sequential decision problem
1. Think of two sequential decision problems and try to formulate them as MDPs. Describe the states, actions and rewards in words.*

One sequential decision problem could be ehether to buy or sell a stock at a given point, or buy a stock at a given moment. This is has 3 actions at each moment, which is buiy, sell and hold. This is the case for each time period, which could be every day or every hour, which is the states. 

2. How do the states, actions and rewards look like for the bandit problem? Try drawing the state-expanded hypergraph.

For the bandit problem, the states are the times between chosing the bandit. Rewards are the reward that are being returned by th bandidts, the payout from from the machines. The actions are chosing the different machines to use.

<img width="415" height="291" alt="image" src="https://github.com/user-attachments/assets/977f0d6d-2cf2-4376-9102-df5a96ae463e" />

## 6.6.2 Exercise - Expected return
1. Suppose $\gamma=0.8$ and we observe the following sequence of rewards: $R_1 = -3$, $R_2 = 5$, $R_3=2$, $R_4 = 7$, and $R_5 = 1$ with a finite time-horizon of $T=5$. What is $G_0$? Hint: work backwards and recall that $G_t = R_{t+1} + \gamma G_{t+1}$.

$G_4=R_5=1$

$G_3=R_4+\gamma G_4=7+0.8\cdot1=7.8$

$G_2=R_3+\gamma G_3 = 2+0.8\cdot7.8=8.24$

$G_1=R_2+\gamma G_2 = 5+0.8\cdot 8.24= 11.592$

$G_0=R_1+\gamma G_1 = -3 + 0.8\cdot 11.592 = 6.2736$

So the total reward is 6.27$.

2. Suppose $\gamma=0.9$ and we observe rewards: $R_1 = 2$, $R_t = 7$, $t>1$ given a infinite time-horizon. What is $G_0$ and $G_1$? Hint: recall that $\sum_{k=0}^\infty x^k = 1/(1-x)$.

$G_1 = \sum_{k=0}^\infty 7 \cdot x^k = 7/(1-x) = 7/1-0.9 = 7/0.1 = 70$

$G_0 = R_1 + \gamma G_1 = 2 + 0.9 \cdot 70 = 65\$$

So the total reward is 65$

## 6.6.3 Exercise - Gambler’s problem

> A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips. The coin may be an unequal coin where there is not equal probability $p_H$ for a head (H) and a tail (T). If the coin comes up heads, the gambler wins as many dollars as he has staked on that flip; if it is tails, he loses his stake. The game ends when the gambler reaches his goal of a capital equal 100, or loses by running out of money. On each flip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars. This problem can be formulated as an undiscounted, episodic, finite MDP, where we assume that the gambler starts with a capital $0 < s_0 < 100$.

1. Define the state space $\mathcal{S}$. Which states are terminal states?

$\mathcal{S} = {0, 1, 2, \cdots , 100}$

The terminal states are S = 0 and S = 100.

2. Define the action space $\mathcal{A}(s)$.

The action space is the number of actions the agent can choose. This is a portion of his capital, so this will be $\mathcal{A}(s) = {0, 1, \cdots, 99}$, as it cannot be 100. 0 might not be included.



