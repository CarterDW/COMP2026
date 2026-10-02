# Chaos and the Butterfly Effect

## The Problem

The Lorenz system is three ODEs with nothing exotic in them:

$$\dot{x} = \sigma(y-x), \qquad \dot{y} = x(\rho-z) - y, \qquad \dot{z} = xy - \beta z,$$

with $\sigma = 10$, $\rho = 28$, $\beta = 8/3$. You already know how to integrate
something like this.

Start two copies a distance $\delta$ apart, as small as you like, and watch them come
apart exponentially. That's the butterfly effect. The rate is the largest Lyapunov
exponent $\lambda$, and it's a number you can go measure this afternoon.

## Goal

Measure $\lambda$. Then work out what your simulation is still good for once you know
the trajectory itself is garbage.

## Natural Steps

1. Integrate it and look at the attractor.
2. Two trajectories, separation $d(t)$, on a log axis.
3. Fit $\lambda$. Think about where you fit it.
4. Find something that *is* reproducible, and show that it is.

## Push Harder

How far ahead can you predict? If errors grow like $e^{\lambda t}$, then improving
your initial data buys you a fixed amount of extra time rather than a proportional
amount. This is why weather forecasts run out after about two weeks instead of getting
steadily longer as instruments improve: the atmosphere has its own positive $\lambda$,
and Lorenz found these equations while working on precisely that problem. Test the
scaling over as many decades of $\delta$ as your machine allows.

There's also an exact check here, which is a rare thing to have in a chaotic system.
The divergence of the flow is $-(\sigma + 1 + \beta)$ at every point in phase space,
so the three Lyapunov exponents have to sum to precisely that. Get the whole spectrum
and check it.
