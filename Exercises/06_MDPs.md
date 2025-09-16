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

The terminal states are S = 0 and S = 100. 0 is a loss and 100 is a win. a single absorbing state might be better. A solution to this problem would be to add a new state 101, which you always go to if you are in state 100 or 0. In this way you have a single absorbing state. It makes it easeier to code, but it is not needed in order to formulate it mathematically. 

2. Define the action space $\mathcal{A}(s)$.

The action space is the number of actions the agent can choose. This is a portion of his capital, so this will be
$\mathcal{A}(s) = \{a\in \mathcal{S}|0< a\le min(s, 100-s)\}$
, as it cannot be 100. 0 might not be included.

3. Let $R_a$ denote the reward given bet $a$ (a stochastic variable). Calculate the expected rewards. If the state-value for the terminal states is set to zero, what do the state-value of a policy mean?

The expected reward can be calculated as:

$r(s,a)=\mathbb{E}[R_a]=a\cdot p_H$

$$
    r(s, a) = \mathbb{E}[R_t | S_{t-1} = s, A_{t-1} = a] = \sum_{r \in \mathcal{R}} r \sum_{s' \in \mathcal{S}} p(s', r | s, a)
$$

where $p_H$ is the chance of getting heads.

The state-value of a policy means that it cannot be made a bet bigger than the capital, which also means there are no options when the capital is 0.

4. Let $R_a$ be zero for all bets $a$ and set the state-value for the terminal state 0 to zero and for state 100 to one. What do the state-value of a policy mean?


## 6.6.4 Exercise - Factory storage

> A factory has a storage tank with a capacity of 4 $\mathrm{m}^{3}$ for temporarily storing waste produced by the factory. Each week the factory produces $0,1$, 2 or 3 $\mathrm{m}^{3}$ waste with respective probabilities 
$$p_{0}=\displaystyle \frac{1}{8},\ p_{1}=\displaystyle \frac{1}{2},\ p_{2}=\displaystyle \frac{1}{4} \text{ and } p_{3}=\displaystyle \frac{1}{8}.$$ 
If the amount of waste produced in one week exceeds the remaining capacity of the tank, the excess is specially removed at a cost of \$30 per cubic metre. At the end of each week there is a regular opportunity to remove all waste from the storage tank at a fixed cost of \$25 and a variable cost of \$5 per cubic metre. 
The problem can be modelled as a finite MDP where a state denote the amount of waste in the tank at the end of week $n$ just before the regular removal opportunity.

1. Define the state space $\mathcal{S}$.

The state space is: 

$\mathcal{S} = \{0, 1, 2, 3, 4\}$

As these are the numbers of cubic meters there can be in the tank at the end of the week, as the excess is removed. It cannot be higher than 4, so five possible states. 

2. Define the action space $\mathcal{A}(s)$.

the actions are: 

$\mathcal{A}(s) = \{\text{remove}, \text{not remove}\}$

It is possible to get the things removed or to do nothing.It could be argued that if the state is s=0, there is nothing to do, as it is never optimal to remove. Otherwise 2 options. 

3. Calculate the expected rewards $r(s, a)$

$r(s,a)=-(\text{cost of empty or cost of special removal})$

There are two different rewards depending if a1 or a2 is taken. 

$r(s, \text{remove}) = -(25 + 5s)$

$$-E(I_{s+i<4}(s+i-4)30)$$

$$r(s, \text{not remove}) = -30\sum_{i>4-s}(s+i-4)\cdot p_i$$

where $i$ is production

4. Calculate the transition probabilities $p(s'|s,a)$

for remove:

$$p(s'|s, \text{remove}) = p_s \text{ for } 0\le s < 4$$

The rest of the probabilities are 0. For not removing:

$$p(4|s, \text{remove}) = \sum_{i\le 4-s} p_i $$
$$p(s'|s,k)=p_{s'-s}\text{ if } s\le s' \le 3$$

if keep:

$s'=4$ if production up to 4m^3^ or more

$(4|s,k)=p(s+i)4)=p(i\ge 4-s)=\sum_{i\ge4-s}p_i$

if s' is less than 4:

$s'=s+i\Rightarrow p(s'|s,k)=p(s'=s+i)=p(i=s'-s)=p_{s'-s}$


otherwise the probability is 0.

