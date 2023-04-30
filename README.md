## Bayesian Experimental Design Framework to Optimize Interferometry Lines of Sight for Future Nuclear Fusion Experiments

Bayesian experimental design has  for interferometry on a simplified model of plasma within a tokamak reactor. This project was produced in collaeration between.   
<img src="https://github.com/jjackson1994/Bayesian_Experimental_Design_for_Fusion/blob/main/images/astat_banner.png?raw=true" alt="astat_banner.png" width=400>

## Key Maths

<a href="#notations" class="markdownIt-Anchor"></a>Notations
------------------------------------------------------------

-   $D$: data of future experiments
-   $\eta$: design parameters
-   $\phi$: parameters of interest (which are relevant for future
    experimental scenarios)
-   $U(D, \eta)$: a utility function to evaluate different
    designs
-   $EU(\eta)$: expected utility
    $EU(\eta) = \int dD \, p(D|\eta) \, U(D, \eta)$

<a href="#important-concepts-and-formulas" class="markdownIt-Anchor"></a>Important concepts and formulas
--------------------------------------------------------------------------------------------------------

-   **Experimental design**: to optimize $EU(\eta)$ with
    respect to $\eta$
-   **Information gain**: given by K-L divergence (how different the
    prior and the posterior are)

$$U_{KL}\left( D,\eta \right) = \int d\phi p\left( \phi\mid D,\eta \right)\log\frac{p\left( \phi\mid D,\eta \right)}{p\left( \phi \right)}$$

-   **Expected utility**:

$$\begin{matrix}
{EU_{KL}\left( \eta \right)} & {= \int dDd\phi p\left( D\mid\eta \right)p\left( \phi\mid D,\eta \right)\log\frac{p\left( \phi\mid D,\eta \right)}{p\left( \phi \right)}} \\
 & {= \int dDd\phi p\left( D\mid\eta \right)\frac{p\left( \phi \right)p\left( D\mid\phi,\eta \right)}{p\left( D\mid\eta \right)}\log\frac{p\left( \phi \right)p\left( D\mid\phi,\eta \right)}{p\left( D\mid\eta \right)p\left( \phi \right)}} \\
 & {= \int dDd\phi p\left( \phi \right)p\left( D\mid\phi,\eta \right)\log\frac{p\left( D\mid\phi,\eta \right)}{p\left( D\mid\eta \right)}} \\
\end{matrix}$$
    
   
where
$p(D|\eta) = \int d\phi \, p(\phi) \, p(D|\phi, \eta)$

<a href="#how-to-calculate-the-integral" class="markdownIt-Anchor"></a>How to calculate the integral?
-----------------------------------------------------------------------------------------------------

**Monte carlo integration**

-   Suppose $p(x)$ is the pdf of random variable $x$
-   The integral of $f(x)p(x)$ can be seen as the
    expected value of a function $f(x)$:

$$\mathbf{E}\left( f\left( x \right) \right) = \int f\left( x \right)p\left( x \right)dx$$

-   By drawing $N$ samples $x_i$ from $p(x)$, the
    integral can be estimated as:

$$\int f\left( x \right)p\left( x \right)dx \approx \frac{1}{N}\sum\limits_{i = 1}^{N}f\left( x_{i} \right)$$

**Calculation of the expected utility for each η** \[1\]

-   First draw a pair of values $(D, \phi)$ from the joint
    distribution
    $p(D, \phi|\eta) = p(\phi) \, p(D|\phi, \eta)$
-   Store the values of $p(D|\phi, \eta)$
-   $p(D|\eta)$ can be estimated as the average of the
    values of $p(D|\phi, \eta)$
-   $EU_{KL} (\eta)$ can be estimated as the average of
    the values of
    $\log\frac{p(D|\phi, \eta)}{p(D|\eta)}$

<a href="#example-interferometry" class="markdownIt-Anchor"></a>Example: interferometry
---------------------------------------------------------------------------------------
-   $D\vec{D}$: line integrated electron density
    ($\vec{D} = \vec{f}(\vec{n}_e)$ where
    $\vec{f}$ is the forward model)
-   $\eta$: geometry of the chords (lines of sight)
-   $\phi$: parameters that determine the (shape of) density profile

<a href="#references" class="markdownIt-Anchor"></a>References
--------------------------------------------------------------

\[1\] Fischer, R. "Bayesian experimental design—studies for fusion
diagnostics." AIP conference proceedings. Vol. 735. No. 1. American
Institute of Physics, 2004.

\[2\] Fischer, R., et al. "Integrated Bayesian experimental design." AIP
Conference Proceedings. Vol. 803. No. 1. American Institute of Physics,
2005.

\[3\] Dreier, H., et al. "Bayesian design of plasma diagnostics." Review
of scientific instruments 77.10 (2006): 10F323.

\[3\] Dreier, H., et al. "Bayesian experimental design of a multichannel
interferometer for Wendelstein 7-X." Review of Scientific Instruments
79.10 (2008): 10E712.
