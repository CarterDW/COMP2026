# Computational Physics

PHYS 7321, Fall 2026

Instructor: Jim Halverson

E-mail: ``j.halverson@northeastern.edu``

TA & Grader: Jacob March ``march.j@northeastern.edu``

# Course Information

**Course Schedule:** Lectures TF 9:50-11:30.

**Course Location:** 315 Behrakis

**Office Hours:** TF 8:45-9:45 in-person, virtual on Slack.

**Course Goal:** Become the best computational physicist you can be. It will suit you well in your research and career.

**Course Format:** In 2026, meeting the course goal means being proficient in both manual and agentic coding. We will format the course accordingly.

Specifically, each week will have a topic, split as:

- **Tuesday** — the *manual session.* A lecture on essentials (~40 min) and associated manual coding portion, no agents (~60 min). This is where you build your coding muscles and essential understanding.
- **Friday** — the *agentic session* (~100 min). You push a more advanced idea within the weekly topic as far as possible, using your deep thinking and coding agents. I will give suggestions, but you are also free to choose your own. Portfolio topics must be of your choosing.

I think the best version of your computational physics self know how to *think* about code, how to *implement essentials*, and how to *use agents* to push the results as a far as possible.

**Specific Goals:**
You'll learn computational techniques applicable to many areas of physics, science more broadly, and STEM industry. Essential skills include:
- Proficiency in Python.
- Proficiency in the basics of ML and AI.
- Assessing problems with respect to the potential use of computational techniques.
- Ability to rapidly develop and implement computational techniques to solve problems.
- Learning to work *with* a coding agent — and, crucially, to audit what it produces.

# Grading

**Portfolio (50%):** At the end of the semester you will submit a portfolio, presented as a slick website (``github.io``) with three projects. The projects are the continuation of three Friday sessions of your choosing.  Each of those Fridays is a **seed**: a starting point
for a piece of deeper computational work, something you're interested in and proud of. Alternatively, if you choose, one of the projects can be a similar difficult project carried out with a potential research advisor on a topic of your choosing. 

**Presentation (10%):** You will have ten minutes to present one of the projects on **December 8**. This is an opportunity for you to tell us something you're excited about!

**Manual-day submissions (20%):** Each manual (Tuesday) session gives you several problems to get hands-on experience with the topic of the week. A subset of the problems from five of those sessions will be submitted to test your understanding. Since these are done during the manual session they are **agent-free**: no LLMs, no autocomplete, just you and the problem. To allow for a hiccup **your lowest score will be dropped,** which can be due to a missed class, should your schedule require it. There are no make-ups. We will wait a few weeks to do one, so you can get the hang of things.

**Participation (20%):** This is a hands-on, in-class course — the manual and agentic
sessions *happen in the room*. Showing up and engaging with the work is a real part of
the grade. I'm looking for attendance, engagement, ``git`` submissions every class, and growth in your abilities throughout the course. To be clear: I am hoping this course will be very fun, so hopefully participation is not an issue! 

# Course Material and Calendar

Details may change, but the current plan for topics are:

**Part 1: Foundations**
- (F) 9/11: Welcome + Python Essentials (Manual)
- (T) 9/15: Neural Networks (Manual)
- (F) 9/18: Neural Networks (Agentic)
- (T) 9/22: Numerical Differentiation and Integration (Manual)
- (F) 9/25: Numerical Differentiation and Integration (Agentic) 

**Part 2: Computational Physics**
- (T) 9/29: Mechanics and Electrostatics (Manual)
- (F) 10/2: Mechanics and Electrostatics (Agentic)
- (T) 10/6: Quantum Mechanics (Manual, submit)
- (F) 10/9: Quantum Mechanics (Agentic)
- (T) 10/13: Statistics and Monte Carlo (Manual)
- (F) 10/16: Statistics and Monte Carlo (Agentic)
- (T) 10/20: Statistical Mechanics and the Ising Model (Manual, submit)
- (F) 10/23: Statistical Mechanics and the Ising Model (Agentic)
- (T) 10/27: Optimization and the Cosmological Constant (Manual)
- (F) 10/30: Optimization and the Cosmological Constant (Agentic)

**Part 3: Machine Learning and Physics**
- (T) 11/3: Supervised Learning (Manual, submit)
- (F) 11/6: Supervised Learning (Agentic)
- (T) 11/10: Convolutions and Galaxy Classification (Manual)
- (F) 11/13: Convolutions and Galaxy Classification (Agentic)
- (T) 11/17: Finding Structure in Data (Manual, submit)
- (F) 11/20: Finding Structure in Data (Agentic)
- (T) 11/24: Reinforcement Learning (Agentic)
- (F) 11/27: **No class** (Thanksgiving / Black Friday)
- (T) 12/1: Generative Models (Manual, submit)
- (F) 12/4: Generative Models (Agentic)
- (T) 12/8: **Presentations** — ten minutes each on one portfolio project

**Finals week (12/14-12/20):** Portfolio due (date TBD).

# Resources

Given the constantly evolving nature of the subject, my lectures will rely on a
number of different resources, including some written by friends: I am indebted to
Fabian Ruehle, Sven Krippendorf, and especially Adrian Feiguin for sharing their
resources. Two excellent resources for the intersection of physics and ML are:
- [IAIFI](http://www.iaifi.org), officially the NSF AI Institute for Artificial Intelligence and Fundamental Interactions, an interdisciplinary institute with a focus on the intersection of AI and physics in collaboration between Northeastern, MIT, Harvard, and Tufts.
- [Physics $\cap$ ML](http://www.physicsmeetsml.org), pronounced "Physics Meets ML," a virtual seminar series organized by myself and some friends that builds on a meeting we ran at Microsoft Research in 2019.

A simple perusal of the website will give you a sense of what's out there. Details can be found in links to slides and videos. More broadly, some canonical ML references are:
- [Deep Learning](https://www.deeplearningbook.org), by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.
- [Geometric Deep Learning](https://geometricdeeplearning.com), by Michael Bronstein, Joan Bruna, Taco Cohen, and Petar Veličković.
- [Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html), by Richard Sutton and Andrew Barto.

Of course, I am also heavily influenced by and indebted to my collaborators on related topics. Some of our recent research works include
- [NN-FT](https://arxiv.org/abs/2307.03223). Neural network field theories are a new approach to field theory inspired by ML theory. The above link presents the most modern and thorough treatment, see also [here](https://arxiv.org/abs/2008.08601) for our original work and [here](https://arxiv.org/abs/2112.04527) for a brief article that developed a number of new concepts, including when an NN-FT is a *quantum* field theory.
- [Searching for Ribbons with Machine Learning](https://arxiv.org/abs/2304.09304) uses ML to build a state of the art algorithm for verifying ribbonness of a knot. Together with slice obstructions, this may be used to study the Smooth Poincaré Conjecture in four dimensions (SPC4), a major open problem in topology. Using it, we ruled out over 800 potential counterexamples to SPC4. See also the fantastic [Man and Machine](https://arxiv.org/abs/0906.5177) article about SPC4.
- [Machine Learning in the String Landscape](https://arxiv.org/abs/1707.00655) one of the original papers on ML for String Theory. One aspect that has stood the test of time is the introduction of conjecture generation, a technique for making ML rigorous.

You might also enjoy lecture notes and articles I've written that cover a number of Physics $\cap$ ML topics:
- [Rigor with Machine Learning from Field Theory to the Poincare Conjecture](https://www.nature.com/articles/s42254-024-00709-0) where some friends and I discuss various ways in which ML techniques can lead to rigorous results.
- [TASI Lectures on Physics for Machine Learning](https://arxiv.org/abs/2408.00082), my perspective (as of 2024) on how physics ideas may inform ML theory through the expressivity, statistics, and dynamics of neural networks.
- [Pre-Strings Lectures on Artificial Intelligence](https://arxiv.org/abs/2607.02905), a more recent (2026) set of lectures on AI aimed at physicists.

# Language Model and Agent Usage

It's 2026 and this is a course on computational physics, so coding agents and large
language models (LLMs) are not just allowed — on Fridays they're the point. My one
rule is that you use *free* tools, for fairness reasons, such as
[ChatGPT](http://chat.openai.com), [GitHub Copilot](https://github.com/features/copilot),
or [Claude Code](http://www.claude.ai) (free with a student account). A goal of this course
is to teach you to be a *productive* computational scientist, and that means using the
best tools available. But — and this is the entire reason for the Tuesday/Friday
split — these tools happily produce code that is wrong or that doesn't run, so you
still have to know what you're doing. If you're just starting out, you may find it
worth going agent-free for a while until you can tell good output from bad. Test it
and see what works.

**A note on agents and honesty:** Use them deliberately — say what you want, read
what comes back, and check it against something you already know. The *analysis*
and the *verification* are yours: a portfolio entry is only worth something if you
can explain the techniques and why you believe the answer.