# Accelerated by Dark Matter: A High-Redshift Pathway to Efficient Galaxy-Scale Star Formation

Authors: Boylan-Kolchin M.

Published: March 2025 ([Link](https://doi.org/10.1093/mnras/staf471))

## Key Ideas

- At fixed halo mass, **gravitational accelerations** within dark matter haloes scale as $(1+z)^2$, meaning haloes at $z \gtrsim 8$ are subject to accelerations $\sim$ 100 times higher than their $z = 0$ counterparts.[^1]
- When the gravitational acceleration exceeds a **critical acceleration** $g_\mathrm{crit} \approx 5 \times 10^{-10}$ m s$^{-2}$ (equivalently $\Sigma_\mathrm{crit} \approx 1000\ M_\odot$ pc$^{-2}$), stellar feedback cannot overcome gravity and star formation becomes *efficient* ($\epsilon \sim 0.5$).[^2]
- Dark matter alone can supply these accelerations on *galactic* scales at high redshift, enabling efficient conversion of baryons into stars across regions far larger than molecular clouds.[^3]
- The **virial mass** corresponding to a fixed virial acceleration scales as $(1+z)^{-6}$, tracking a roughly constant cumulative comoving number density of $n(>M_\mathrm{vir}) \approx 10^{-4}$ Mpc$^{-3}$ at $z \gtrsim 8$.[^4]
- For somewhat rarer haloes, the model predicts stellar masses $M_\star \sim 10^9\ M_\odot$ forming in regions of $\mathcal{O}(100\ \mathrm{pc})$ over $\sim$ 40 Myr time-scales at $z \approx 12$–$14$, consistent with JWST observations of the brightest galaxies at that epoch.[^5]

## Introduction

- Star formation is generally inefficient ($\epsilon_\mathrm{ff} \lesssim 2\%$), regulated by stellar feedback, except in regions where gravitational accelerations exceed $g_\mathrm{crit} \approx \dot{p}/m \approx 5 \times 10^{-10}$ m s$^{-2}$, at which point feedback becomes insufficient and star formation efficiency approaches order unity.[^6]
- In the local Universe, only dense baryonic concentrations (molecular cloud cores) reach these accelerations; dark matter haloes at $z = 0$ have central accelerations well below $g_\mathrm{crit}$ (e.g. the Milky Way reaches $g_\mathrm{max}/G \approx 500\ M_\odot$ pc$^{-2}$, compared to $g_\mathrm{crit}/G = 3100\ M_\odot$ pc$^{-2}$).[^7]
- JWST has revealed unexpectedly active galaxy and black hole growth at cosmic dawn, motivating the question of whether high dark matter densities at early times can naturally explain efficient galaxy-wide star formation.[^8]

## Data

- This is a purely theoretical/analytical paper; no new observational data were generated or analysed.[^9]
- Comparison is made to spectroscopically confirmed high-redshift galaxies from JWST, including **GN-z11**, **GS-z14**, **GHZ12**, **Maisie’s galaxy**, **GS-z9**, and **RUBIES-EGS-55604/966323**, drawn from the CEERS and JADES surveys.[^10]

## Method

- The author derives the acceleration profile of dark matter haloes assuming an **NFW profile**, where the acceleration at radius $\tilde{r} = r/R_\mathrm{vir}$ is $g(r) = g_\mathrm{vir}\ \mu(c)/[\mu(c\tilde{r})\ \tilde{r}^2]$, with $\mu(x) = \ln(1+x) - x/(1+x)$.[^11]
- The **virial acceleration** is shown to scale as $g_\mathrm{vir} \propto M_\mathrm{vir}^{1/3}(1+z)^2$, meaning one can invert this to find the halo mass at which a given acceleration is reached: $M_\mathrm{vir} \propto (g_\mathrm{vir}/G)^3 (1+z)^{-6}$.[^12]
- Halo concentrations are drawn from the median $c(M_\mathrm{vir}|z)$ relations of Ishiyama et al. (2021) at $z = 0$ and the Yung et al. (2024) simulations at $z = 10$, implemented via **COLOSSUS**; the **halo mass function** is computed using the **HMF** code to determine cumulative number densities.[^13]

## Results

- Haloes at fixed cumulative comoving number density have nearly constant virial accelerations at $z \gtrsim 8$; the threshold virial acceleration for efficient star formation ($g_\mathrm{max} = g_\mathrm{crit}$ at the halo centre) corresponds to $g_\mathrm{vir}/G \approx 380\ M_\odot$ pc$^{-2}$ and traces $n(>M_\mathrm{vir}) \approx 10^{-4}$ Mpc$^{-3}$ from $z = 20$ to $z = 8$.[^14]
- Assuming baryons trace the dark matter distribution ($c_b = 1$) with an integrated star formation efficiency $\epsilon = 0.5$, the model predicts a star formation rate of $\dot{M}_\star = 24\ M_\odot$ yr$^{-1}$ and an unattenuated UV magnitude of $M_\mathrm{UV} = -21.7$ for $M_\star = 10^9\ M_\odot$, with a specific star formation rate scaling as $M_\star^{-1/4}$.[^15]
- These predictions agree well with JWST observations of **GN-z11** ($M_\star \approx 10^9\ M_\odot$, $r_{1/2} = 64$ pc, $\dot{M}_\star \approx 20\ M_\odot$ yr$^{-1}$, $M_\mathrm{UV} = -21.6$), and the implied halo number densities for the observed high-redshift galaxy sample are consistent with the volumes surveyed by CEERS and JADES ($V \approx$ a few $\times 10^5$ Mpc$^3$).[^16]

## Discussion

- The redshift dependence of the threshold mass for efficient star formation is remarkably similar to that predicted by the **feedback-free burst** (FFB) model of Dekel et al. (2023), despite relying on different physical assumptions; however, this model predicts an order of magnitude *lower* stellar masses and a factor of 2–3 lower star formation rates at fixed halo mass.[^17]
- In the high-acceleration regime, stars should form preferentially in self-bound clusters, consistent with JWST observations of lensed systems dominated by young star clusters; baryonic masses of $\sim 10^6\ M_\odot$ at high accelerations as early as $z \approx 20$ are intriguingly similar to globular cluster mass scales.[^18]
- The model predicts that massive rare haloes ($n \approx 10^{-8}$ Mpc$^{-3}$) may undergo repeated cycles of efficient starbursts with duty cycles of $\sim$ 35%, whilst more common haloes ($n \approx 10^{-5}$ Mpc$^{-3}$) have duty cycles of at most $\sim$ 15%.[^19]

## Weaknesses

- The model assumes baryons trace the dark matter spatial distribution (i.e. $c_b = 1$), which provides only an *upper limit* on the baryonic mass at high accelerations; in reality, gas physics (cooling, pressure support) will modify this distribution.[^20]
- The precise value of $g_\mathrm{crit}$ is treated as a single threshold rather than a smooth transition, and variations in the **stellar initial mass function** (IMF) at high redshift (e.g. a top-heavy IMF) could alter $g_\mathrm{crit}$ via changes in the momentum flux $\dot{p}/m$.[^21]
- The paper does not fold in the full cosmological distribution of halo concentrations at fixed mass and redshift, relying instead on median concentration–mass relations; scatter in concentration would affect the predicted mass above $g_\mathrm{crit}$.[^22]

## Conclusions

- Dark matter haloes at $z \gtrsim 8$ naturally produce gravitational accelerations exceeding $g_\mathrm{crit}$ in their central regions, enabling efficient star formation on galactic scales — a regime that is *not* accessible at low redshift, where only dense baryonic cores of molecular clouds can reach comparable accelerations.[^23]
- The threshold virial mass for this process tracks $n(>M_\mathrm{vir}) \approx 10^{-4}$ Mpc$^{-3}$ and the baryonic mass available for efficient conversion into stars can reach $\sim 10^9$–$10^{10}\ M_\odot$ in sufficiently rare haloes, in good agreement with JWST measurements of bright high-redshift galaxies.[^24]
- Any significant population of galaxies requiring $M_\star/(f_b\ M_\mathrm{halo}) \approx 1$ would *remain* difficult to explain within $\Lambda$CDM, as even this mechanism cannot convert virtually all halo baryons into stars.[^25]

## Future Work

- A more detailed comparison with the feedback-free burst model of Dekel et al. (2023) is needed to determine whether the two frameworks are in conflict or concordance.[^26]
- Incorporating a full $\epsilon$–$g$ relation (rather than a single threshold) and the cosmological scatter in halo concentrations at fixed mass would refine quantitative predictions.[^27]
- Understanding the fate of these high-density regions at lower redshifts — whether dark matter densities decrease through hierarchical assembly or persist as high-concentration haloes/substructure at $z = 0$ — is a key avenue for future investigation.[^28]

## Glossary

| Term                                                | Definition                                                                                                                                   |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| NFW profile                                         | Navarro–Frenk–White density profile for dark matter haloes: $\rho(r) \propto r^{-1}(1 + r/r_s)^{-2}$                                         |
| Virial radius ($R_\mathrm{vir}$)                    | Radius enclosing a mean overdensity $\Delta_\mathrm{vir}(z)\ \rho_m(z)$ relative to the background matter density                            |
| Virial acceleration ($g_\mathrm{vir}$)              | Gravitational acceleration at the virial radius: $g_\mathrm{vir} = G M_\mathrm{vir}/R_\mathrm{vir}^2$                                        |
| Critical acceleration ($g_\mathrm{crit}$)           | Acceleration above which stellar feedback cannot overcome gravity ($\approx 5 \times 10^{-10}$ m s$^{-2}$)                                   |
| Critical surface density ($\Sigma_\mathrm{crit}$)   | Surface mass density equivalent of $g_\mathrm{crit}$: $\Sigma_\mathrm{crit} = g_\mathrm{crit}/(\pi G) \approx 1000\ M_\odot$ pc$^{-2}$       |
| Halo concentration ($c$)                            | Ratio of virial radius to NFW scale radius: $c = R_\mathrm{vir}/r_s$                                                                         |
| Star formation efficiency ($\epsilon_\mathrm{ff}$)  | Fraction of gas converted into stars per free-fall time                                                                                      |
| Integrated star formation efficiency ($\epsilon$)   | Total fraction of baryons within the high-acceleration region converted into stars ($= \eta_\mathrm{ff}\ \epsilon_\mathrm{ff}$)              |
| Free-fall time ($t_\mathrm{ff}$)                    | Characteristic gravitational collapse time-scale at mean density within $r_\mathrm{crit}$                                                    |
| Feedback-free burst (FFB)                           | Model of Dekel et al. (2023) positing highly efficient star formation in massive haloes before supernova feedback can act                    |
| Cosmic baryon fraction ($f_b$)                      | Universal ratio of baryon to total matter density: $f_b = \Omega_b/\Omega_m = 0.156$                                                         |
| Threshold virial acceleration ($g_\mathrm{thresh}$) | Minimum virial acceleration for a halo’s centre to reach $g_\mathrm{crit}$; corresponds to $g_\mathrm{vir}/G \approx 380\ M_\odot$ pc$^{-2}$ |
| Cumulative number density ($n(>M_\mathrm{vir})$)    | Comoving number density of haloes more massive than $M_\mathrm{vir}$                                                                         |

## Tags

#COLOSSUS #HMF #JWST #CEERS #JADES #NumPy #SciPy #Matplotlib
#GalaxiesFormation #GalaxiesHighRedshift #GalaxiesStarFormation #CosmologyDarkMatter #CosmologyTheory #CosmologyDarkAgesReionizationFirstStars #Gravitation #StarsFormation #GalaxiesStarburst #GalaxiesHaloes

## References

[^1]: “at fixed halo mass, the acceleration at the virial radius scales roughly as $(1 + z)^2$, which means haloes of a fixed mass are subject to accelerations $\sim$100 times higher at $z \approx 10$ than at $z = 0$.” (Section 1, p.3210)
[^2]: “An important exception is dense regions where baryons experience high accelerations: in this case, momentum injection from massive stars, $\dot{p}/m$, is insufficient to overcome gravity and star formation becomes efficient: $M_\star = \epsilon\ M_\mathrm{gas}$, with $\epsilon \sim \mathcal{O}(1)$.” (Section 1, p.3210)
[^3]: “I point out in this paper that at high redshift, the significantly higher mean density of the Universe results in regions within galaxy–mass haloes where dark matter can provide the necessary accelerations for efficient formation of galaxy-scale quantities of stars ($M_\star \sim 10^8$–$10^{10}\ M_\odot$).” (Section 5, p.3217)
[^4]: “The virial mass resulting in a fixed virial acceleration scales as $(1 + z)^{-6}$.” (Section 3.1, p.3212) and “it closely follows the contour for a constant cumulative comoving number density of $n \approx 10^{-4}$ Mpc$^{-3}$ from $z = 20$ to 8.” (Section 3.1, p.3212)
[^5]: “For somewhat rarer haloes, this model predicts stellar masses of $M_\star \sim 10^9\ M_\odot$ can form in regions that end up with sizes $\mathcal{O}(100\ \mathrm{pc})$ over 40 Myr time-scales at $z \approx 12$–$14$; these numbers compare well to measurements for some of the brightest galaxies at that epoch from JWST observations.” (Abstract, p.3210)
[^6]: “Star formation is generally regulated by stellar feedback: young, massive stars have prodigious UV output, leading to a variety of physical mechanisms that inhibit further star formation. The star formation efficiency $\epsilon_\mathrm{ff}$ – the fraction of gas converted into stars on a free-fall time – is therefore low, typically $\lesssim 2$ per cent, even in molecular clouds.” (Section 1, p.3210)
[^7]: “The Milky Way ($M_\mathrm{vir} = 10^{12}\ M_\odot$, $c \approx 10$) has $g_\mathrm{vir}(z = 0)/G \approx 14\ M_\odot$ pc$^{-2}$, so the maximum acceleration from dark matter is $g_\mathrm{max}/G \approx 500\ M_\odot$ pc$^{-2}$ or $g_\mathrm{max} \approx 7 \times 10^{-11}$ m s$^{-2}$, well below $g_\mathrm{crit}$.” (Section 3.1, p.3212)
[^8]: “Galaxy formation efficiency at these redshifts has recently shifted from purely theoretical speculation to an urgent observational and theoretical puzzle. JWST has revealed an epoch of strikingly and unexpectedly active galaxy and black hole growth at cosmic dawn.” (Section 1, p.3210)
[^9]: “No new data were generated or analysed in support of this research.” (Data Availability, p.3217)
[^10]: “Several high-redshift galaxies with redshifts that have been spectroscopically confirmed by JWST are also shown on the plot. These galaxies have inferred stellar masses that are consistent with the predictions of efficient galaxy formation in the high (dark matter) acceleration regime, as they lie close to the volumes surveyed by CEERS and JADES of $V \approx$ (a few) $\times 10^5$ Mpc$^3$.” (Section 3.4, p.3215)
[^11]: “for a Navarro, Frenk & White (1996, 1997, hereafter, NFW) profile, the acceleration at radius $\tilde{r} \equiv r/R_\mathrm{vir}$ depends only on the virial acceleration and the halo concentration: $g(r) = g_\mathrm{vir}\ \mu(c)/[\mu(c\tilde{r})\ \tilde{r}^2]$, where $\mu(x) \equiv \ln(1 + x) - x/(1 + x)$.” (Section 2, p.3211)
[^12]: “The virial acceleration therefore increases as $(1 + z)^2$.” (Section 3.1, p.3212) and “The virial mass resulting in a fixed virial acceleration scales as $(1 + z)^{-6}$.” (Section 3.1, p.3212)
[^13]: “I assume the mean $c(M|z = 0)$ relation from Ishiyama et al. (2021) as implemented in COLOSSUS (Diemer 2018); at $z = 10$, the shaded region corresponds to concentrations between $c = 2$ and $c = 5.5$, which approximately spans the symmetric 68 per cent interval around the median concentration found in the cosmological simulations of Yung et al. (2024).” (Section 3.1, p.3212)
[^14]: “This threshold is not a constant value with redshift because halo concentrations at a given $M_\mathrm{vir}$ evolve somewhat with time, but the figure shows this evolution has a very minor effect on $g_\mathrm{thresh}$: it closely follows the contour for a constant cumulative comoving number density of $n \approx 10^{-4}$ Mpc$^{-3}$ from $z = 20$ to 8.” (Section 3.1, p.3212)
[^15]: “Using equation (11), the star formation rate can be written as $\dot{M}_\star = 24\ M_\odot$ yr$^{-1}$” (Section 3.3, p.3214) and “$M_\mathrm{UV,un} = -21.7 - 1.875\ \log_{10}[M_\star/(10^9\ M_\odot)]$” (Section 3.3, p.3215)
[^16]: “for example, Tacchella et al. (2023) find that GN-z11 has a stellar mass of $\approx 10^9\ M_\odot$ within a half-light radius of 64 pc and a star formation rate of $\approx 20\ M_\odot$ yr$^{-1}$. All of these values, as well as its observed $M_\mathrm{UV}$ value of $-21.6$ mag with 0.2 mag of obscuration, agree very well with the expectations of the model described here.” (Section 3.4, p.3215)
[^17]: “Dekel et al. (2023) quote an expected stellar mass of $M_\star \approx 10^{10}\ M_\odot$ at $z \approx 10$ in haloes of $M_\mathrm{vir} \approx 10^{10.8}\ M_\odot$ with a star formation rate of 65 $M_\odot$ yr$^{-1}$; at the same halo mass, the model described here would result in an order of magnitude lower stellar mass as well as a star formation rate that is lower by a factor of 2–3.” (Section 4, p.3216)
[^18]: “At the high accelerations considered here, the high efficiency of star formation is not the only expected change: stars should form preferentially in self-bound clusters.” (Section 4, p.3216) and “the mass in baryons above $g_\mathrm{crit}$ in the lowest-mass (and therefore most common) haloes achieving $g_\mathrm{thresh}$ at their centres is $\sim 10^6\ M_\odot$, a mass scale intriguingly similar to that of globular clusters, at all redshifts.” (Section 4, p.3216)
[^19]: “very rare and massive haloes of $n(>M_\mathrm{vir}) \approx 10^{-8}$ Mpc$^{-3}$ may go through cycles of efficient bursts with a duty cycle of $\sim 35$ per cent, whereas haloes with $n(>M_\mathrm{vir}) \approx 10^{-5}$ Mpc$^{-3}$ will have duty cycles of at most 15 per cent.” (Section 4, p.3216)
[^20]: “For the purposes of this plot, I assume that each halo has its cosmic fraction of baryons, $M_b(< R_\mathrm{vir}) = f_b\ M_\mathrm{vir}$, and that the baryons have a spatial distribution matching that of the dark matter. This gives an upper limit to the stellar mass content of a halo that can come from efficient conversion of baryons in a dark matter halo via the mechanism described here.” (Section 3.2, p.3213)
[^21]: “if the stellar IMF differs substantially in bright systems at early cosmic times relative to lower redshifts, $g_\mathrm{crit}$ could vary as well: for example, a more top-heavy IMF, as has been invoked to explain JWST observations, would increase $\dot{p}/m$ and therefore $g_\mathrm{crit}$.” (Section 4, p.3216)
[^22]: “Future avenues for exploration include folding in a full $\epsilon$–$g$ relation as described in, e.g. Fall et al. (2010), Grudić et al. (2018), or Hopkins et al. (2022) and a cosmological distribution of concentrations at fixed halo mass and redshift (as the central gravitational acceleration at a given halo mass and redshift depends only on concentration via equation (4)).” (Section 5, p.3217)
[^23]: “While star formation is generally inefficient when considered as the fraction of gas turned into stars on a local dynamical time or integrated over the lifetime of a star-forming region, efficient star formation can happen when stellar feedback cannot overcome the gravity of star-forming gas.” (Section 5, p.3217)
[^24]: “The amount of baryonic mass contained in the region of high acceleration is $\approx 10^6\ M_\odot$ at the threshold mass; for more massive (and therefore rarer) haloes, the mass in baryons subject to high accelerations can be comparable to the observed masses of the highest redshift galaxies.” (Section 5, p.3217)
[^25]: “Any significant population of galaxies that require integrated star formation efficiencies of $M_\star/(f_b\ M_\mathrm{halo}) \approx 1$ would remain very difficult to understand within $\Lambda$CDM.” (Section 4, p.3216)
[^26]: “An avenue of future interest is a more detailed comparison of the two models and an exploration of whether their predictions are in conflict or concordance.” (Section 4, p.3216)
[^27]: “Future avenues for exploration include folding in a full $\epsilon$–$g$ relation as described in, e.g. Fall et al. (2010), Grudić et al. (2018), or Hopkins et al. (2022) and a cosmological distribution of concentrations at fixed halo mass and redshift.” (Section 5, p.3217)
[^28]: “Understanding the fate of the predicted regions of high galaxy formation efficiency will also be important, as within the basic paradigm described in this paper, they must either become less dense with time or represent the high-concentration tail of massive halos (or their substructure) in the local Universe.” (Section 5, p.3217)