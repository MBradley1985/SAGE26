# SAGE26 Paper II: scaling relations rederived

Every scaling relation and derived number in the Paper II (FFB) draft of 5 October 2026, rederived from its assumptions. Numbers use the model's own conventions and code (`plotting/paper_plots.py`) and can be reproduced from the formulae below. A standalone LaTeX version is in `scaling_relations.tex`.

Status key: **verified**, **needs qualification**, **incorrect**. Every flagged item has a numbered fix (F1–F11) with replacement text in the last section, [Corrections](#corrections-replacement-text-for-the-draft).

## Summary

| # | Draft statement | Rederived | Status | Fix |
|---|---|---|---|---|
| R1 | $M_{\rm vir,FFB}=10^{10.8}\,M_\odot\,[(1+z)/10]^{-6.2}$; $3.4\times10^{10}$ at $z=10.1$ | $3.35\times10^{10}$ | verified | — |
| R2 | sigmoid, $\Delta\log M=0.15$ | SD $0.272$ dex, central slope $1.67$ dex$^{-1}$ | verified | — |
| R3 | quench mass $10^{12}$ gives floor $z>6$ | floor $z=5.40$ | needs qualification | F5 |
| R4 | $g_{\rm crit}=4.3\times10^{-10}$ m s$^{-2}$, $\Sigma_{\rm crit}=987$ | $4.32\times10^{-10}$, $987$ | verified | — |
| R5 | $g_{\max}=g_{\rm vir}c^2/[2\mu(c)]$ | derived (central limit) | verified | — |
| R6 | $g_{\rm vir}/G=310$–$440$ ($c=3$–$4$), $150$ ($c=7$) | $314$–$438$, $152$ | verified | — |
| R7 | $g_{\rm vir}\propto M^{1/3}(1+z)^2$; "mean halo density tracks critical density" | exact form below; needs matter domination | verified, reword | F1 |
| R8 | $M_{\rm DMESF}\propto(1+z)^{-6}$ at fixed $c$ | slope $-6\,\Omega_m(z)$ = $-5.95$ to $-5.99$ | verified | — |
| R9 | $M_{\rm DMESF}\propto(1+z)^{-6.16}$ | $-6.09$ ($6<z<12$); range-dependent $-6.06$ to $-6.21$ | incorrect | F1, F2 |
| R9 | $10^{10.85}$ at $z\approx9$, $0.04$ dex above FFB | $10^{10.845}$, $+0.045$ dex | verified | — |
| R9 | max $0.047$, RMS $0.039$ dex "every median in Figure 5" | values right, but they are threshold-curve differences | needs qualification | F7 |
| R10 | concentration scatter spreads threshold by $0.3$ dex | $0.30$ dex per halo; population width $0.235$ dex | needs qualification | F1 |
| R10 | $\sigma(\log M)\approx1.17\,\sigma_{\ln c}$ "since $M^{1/3}\propto\mu/c^2$" | $1.18$ is right but needs the $\bar c(M)$ slope; the stated reason gives $1.48$ | needs qualification | F3 |
| R11 | $\Delta\log M=0.15$ ↔ $\sigma_{\ln c}=0.18$, agree to 10% | $0.23$ (SD match, 15%) or $0.20$ (slope match, 2%) | incorrect | F3 |
| R12 | MW $g_{\max}/G\approx576$, cluster $\approx2123$ | inputs unstated; BK25 500; model convention 684 / 2338 | needs qualification | F4 |
| R12 | "no halo in the local Universe attains $g_{\rm crit}$" | 13% of $10^{15}\,M_\odot$ clusters do | incorrect | F4 |
| R13 | $10^9\,M_\odot$ haloes reach it by $z\approx18$ | $10^{9.04}$ at $z=18$ | verified | — |
| R14 | $r_{\rm disk}=(3/\sqrt2)\lambda R_{\rm vir}$ | $=3R_d$ (MMW98); $t_{\rm dyn}=3\lambda/(\sqrt\Delta H)$ | verified | — |
| R15 | $t_{\rm ff}\sim1$ Myr at $n\sim3\times10^3$ cm$^{-3}$ | $0.79$–$0.94$ Myr | verified | — |
| R16 | $\epsilon_{\rm SFE}$ "approaching the assumed maximum of $\alpha_{\rm eff}=0.2$" | $\alpha_{\rm eff}$ does not bound $\epsilon_{\rm SFE}$ | needs qualification | F6 |

## Conventions and assumptions

- **A1 Cosmology.** Flat ΛCDM, Millennium (WMAP1): $\Omega_m=0.25$, $\Omega_\Lambda=0.75$, $h=0.73$.
$$H(z)=H_0E(z),\qquad E^2(z)=\Omega_m(1+z)^3+\Omega_\Lambda,\qquad \rho_c(z)=\frac{3H^2(z)}{8\pi G}$$
- **A2 Halo definition.** Spherical overdensity at 200 times critical, as in SAGE:
$$M_{\rm vir}=\frac{4\pi}{3}\,\Delta\,\rho_c(z)\,R_{\rm vir}^3,\qquad \Delta=200$$
- **A3 NFW profile**, $c\equiv R_{\rm vir}/r_s$:
$$M(<r)=M_{\rm vir}\frac{\mu(r/r_s)}{\mu(c)},\qquad \mu(x)=\ln(1+x)-\frac{x}{1+x}$$
- **A4 Concentration.** Median $\bar c(M,z)$ from Ishiyama et al. (2021), 200c, via `colossus`; log-normal scatter $\ln c=\ln\bar c+\sigma_{\ln c}\zeta$, $\zeta\sim\mathcal N(0,1)$, $\sigma_{\ln c}=0.2$.
- **A5** $g_{\rm crit}/G=3100\,M_\odot\,{\rm pc}^{-2}$ (Boylan-Kolchin 2025).
- **A6** Li et al. (2024): normalisation $10^{10.8}\,M_\odot$ at $z=9$, slope $-6.2$, sigmoid width $\Delta\log M=0.15$ dex.
- **A7 Constants.** $G=6.674\times10^{-11}$ m$^3$ kg$^{-1}$ s$^{-2}$, $M_\odot=1.989\times10^{30}$ kg, 1 pc $=3.0857\times10^{16}$ m. Masses are physical $M_\odot$ (no $h$).

## FFB threshold (Li+24)

### R1. Threshold mass (draft Eq. 1): verified

$$\log_{10}\frac{M_{\rm vir,FFB}(z)}{M_\odot}=10.8-6.2\,\log_{10}\frac{1+z}{10}$$

Adopted from Dekel et al. (2023) and Li et al. (2024), not derived here. At $z=9$ it is exactly $10^{10.8}\,M_\odot$; at the $z=10.07$ snapshot it is $10^{10.525}=3.35\times10^{10}\,M_\odot$ (draft $3.4\times10^{10}$). The code works in $10^{10}h^{-1}M_\odot$ and the $h$ cancels on conversion.

### R2. Sigmoid transition (draft Eq. 2): verified

With $y=\log_{10}M_{\rm vir}$ and $y_0=\log_{10}M_{\rm vir,FFB}$:

$$f_{\rm FFB}=\frac{1}{1+e^{-(y-y_0)/\Delta\log M}}$$

This is the cumulative distribution of a logistic variable of scale $\Delta\log M$:

$$\sigma_{\rm logistic}=\frac{\pi\,\Delta\log M}{\sqrt3}=0.272~{\rm dex},\qquad \left.\frac{df_{\rm FFB}}{dy}\right|_{y_0}=\frac{1}{4\,\Delta\log M}=1.67~{\rm dex}^{-1}$$

### R3. Redshift floor from the quenching mass: needs qualification

**Fix:** F5.

$$\frac{1+z_q}{10}=10^{-(12-10.8)/6.2}\quad\Rightarrow\quad z_q=10^{\,1-1.2/6.2}-1=5.40$$

The draft says the quenching mass gives "a redshift floor of $z>6$". The threshold midpoint reaches $10^{12}\,M_\odot$ at $z=5.4$. Quote $z\approx5.4$ or cite where Li+24 place the floor at $z\approx6$.

## Critical acceleration (BK25)

### R4. $g_{\rm crit}$ and $\Sigma_{\rm crit}$: verified

$$g_{\rm crit}=G\times3100\,M_\odot\,{\rm pc}^{-2}=4.32\times10^{-10}~{\rm m\,s^{-2}},\qquad \Sigma_{\rm crit}\equiv\frac{g_{\rm crit}}{\pi G}=\frac{3100}{\pi}=987\,M_\odot\,{\rm pc}^{-2}$$

## Maximum acceleration of an NFW halo

### R5. $g_{\max}=g_{\rm vir}c^2/[2\mu(c)]$ (draft Eqs. 3–4): verified

With $x=r/r_s=c\,r/R_{\rm vir}$:

$$g(r)=\frac{GM(<r)}{r^2}=\frac{GM_{\rm vir}}{R_{\rm vir}^2}\,\frac{c^2}{\mu(c)}\,\frac{\mu(x)}{x^2}=g_{\rm vir}\,\frac{c^2}{\mu(c)}\,\frac{\mu(x)}{x^2},\qquad g_{\rm vir}\equiv\frac{GM_{\rm vir}}{R_{\rm vir}^2}$$

$\mu(x)/x^2$ decreases monotonically. With $\mu'(x)=x/(1+x)^2$,

$$\frac{d}{dx}\frac{\mu}{x^2}=\frac{x\mu'-2\mu}{x^3},\qquad \frac{d}{dx}\left(x\mu'-2\mu\right)=-\frac{2x^2}{(1+x)^3}<0,$$

and $x\mu'-2\mu=0$ at $x=0$, so it is negative for all $x>0$. The maximum is the central limit. Since $\mu(x)=x^2/2-2x^3/3+O(x^4)$, $\mu(x)/x^2\to1/2$:

$$g_{\max}=\lim_{r\to0}g(r)=g_{\rm vir}\,\frac{c^2}{2\,\mu(c)}$$

*Assumptions:* a pure NFW cusp to $r\to0$, baryons neglected. The central acceleration is finite although the density diverges.

### R6. Threshold virial acceleration: verified

$$g_{\rm vir,thr}=g_{\rm crit}\,\frac{2\mu(c)}{c^2}$$

| $c$ | 3 | 3.2 | 4 | 5 | 7 |
|---|---|---|---|---|---|
| $2\mu(c)/c^2$ | 0.1414 | 0.1315 | 0.1012 | 0.0767 | 0.0492 |
| $g_{\rm vir,thr}/G$ [$M_\odot$ pc$^{-2}$] | 438 | 408 | 314 | 238 | 152 |

## Virial scalings and the threshold mass

### R7. Virial radius, acceleration and surface density: verified (reword)

**Fix:** F1.

From A1–A2, $M_{\rm vir}=\Delta H^2R_{\rm vir}^3/(2G)$, so

$$R_{\rm vir}=\left(\frac{2GM_{\rm vir}}{\Delta H^2(z)}\right)^{1/3},\qquad g_{\rm vir}=(GM_{\rm vir})^{1/3}\left(\frac{\Delta H^2(z)}{2}\right)^{2/3},\qquad \Sigma_{\rm vir}\equiv\frac{M_{\rm vir}}{\pi R_{\rm vir}^2}=\frac{g_{\rm vir}}{\pi G}$$

The virial surface density is the virial acceleration in the units of $\Sigma_{\rm crit}=g_{\rm crit}/(\pi G)$, so the two thresholds are the same condition. When matter dominates, $H^2\simeq H_0^2\Omega_m(1+z)^3$, and

$$R_{\rm vir}\propto M_{\rm vir}^{1/3}(1+z)^{-1},\qquad g_{\rm vir}\propto M_{\rm vir}^{1/3}(1+z)^{2},\qquad \Sigma_{\rm vir}\propto M_{\rm vir}^{1/3}(1+z)^{2}$$

*Wording.* With a 200c definition, "the mean halo density tracks the critical density" is true by construction at every $z$. The $(1+z)$ scalings need the separate assumption $\Omega_m(z)\equiv\Omega_m(1+z)^3/E^2\approx1$, which holds well here: $\Omega_m(z)=0.977,\,0.991,\,0.997,\,0.999$ at $z=4,6,9,12$.

### R8. Closed-form threshold mass at fixed concentration: verified

$$M_{\rm vir,DMESF}(z;c)=\frac{4}{G\,\Delta^2H^4(z)}\left[g_{\rm crit}\,\frac{2\mu(c)}{c^2}\right]^3$$

This matches `mbk25_threshold_mass_msun` to $10^{-4}$ dex: $10^{10.845}\,M_\odot$ at $z=9$, $c=3.20$; $10^{10.572}=3.73\times10^{10}\,M_\odot$ at $z=10.07$, $c=3.22$ (draft $3.7\times10^{10}$). At fixed $c$:

$$\left.\frac{d\ln M_{\rm vir,DMESF}}{d\ln(1+z)}\right|_c=-4\,\frac{d\ln H}{d\ln(1+z)}=-6\,\Omega_m(z)$$

That is $-5.86,\,-5.95,\,-5.98,\,-5.99$ at $z=4,6,9,12$, tending to $-6$. A fit at $c=3.2$ over $6<z<12$ gives $-5.98$.

### R9. Median-concentration threshold and its exponent: incorrect ($-6.16$)

**Fix:** F1 and F2 (exponent), F7 (0.047 dex attribution).

With $c=\bar c(M,z)$:

$$\ln M={\rm const}-4\ln H(z)+3\ln\frac{\mu(\bar c)}{\bar c^2}$$

$$\kappa(c)\equiv\frac{d\ln(\mu/c^2)}{d\ln c}=\frac{c^2}{(1+c)^2\mu(c)}-2,\qquad s_M\equiv\frac{\partial\ln\bar c}{\partial\ln M},\qquad s_z\equiv\frac{\partial\ln\bar c}{\partial\ln(1+z)}$$

$$\frac{d\ln M_{\rm vir,DMESF}}{d\ln(1+z)}=\frac{-6\,\Omega_m(z)+3\kappa\,s_z}{1-3\kappa\,s_M}$$

At $z=9$ ($\bar c=3.20$): $\kappa=-1.138$, $s_M=+0.076$, $s_z=+0.507$, giving a local slope of $-6.12$. The median concentration rises with redshift at fixed mass ($s_z>0$), which steepens the slope; its rise with mass ($s_M>0$) partly offsets this.

| fit range | $4<z<15$ | $5<z<13$ | $6<z<12$ | $8<z<12$ |
|---|---|---|---|---|
| exponent | $-6.06$ | $-6.06$ | $-6.09$ | $-6.21$ |

$-6.16$ is not recovered over $6<z<12$, the range used for the residuals. Quote $\approx-6.1$ and state the range. At $z=9$, $\log_{10}M_{\rm vir,DMESF}=10.845$, $+0.045$ dex above Li+24 (verified). Over $6<z<12$ the two threshold curves differ by at most $0.047$ dex, RMS $0.039$ dex. Those numbers are right, but they are differences between the threshold curves of Figure 1, not between "every median in Figure 5".

### R10. Threshold-mass scatter from concentration scatter: needs qualification

**Fix:** F1 (0.3 dex), F3 (coefficient).

A halo with quantile $\zeta$ sits on the threshold when

$$\tfrac13\ln M+\ln\frac{c^2}{2\mu(c)}={\rm const}(z),\qquad \ln c=\ln\bar c(M,z)+\sigma_{\ln c}\,\zeta$$

Since $d\ln[c^2/2\mu]/d\ln c=-\kappa$:

$$\tfrac13\,d\ln M-\kappa\left(s_M\,d\ln M+\sigma_{\ln c}\,d\zeta\right)=0\quad\Rightarrow\quad \frac{d\ln M_t}{d\zeta}=\frac{\kappa\,\sigma_{\ln c}}{\tfrac13-\kappa s_M}$$

Two widths follow, and the draft uses both:

$$\text{(a) one halo, }\bar c\text{ fixed:}\quad \frac{\partial\log_{10}M_t}{\partial\ln c}=\frac{3|\kappa|}{\ln10}=1.48$$

$$\text{(b) population width:}\quad \sigma(\log_{10}M_t)=\frac{|\kappa|}{\ln10\,(\tfrac13-\kappa s_M)}\,\sigma_{\ln c}=1.18\,\sigma_{\ln c}\ (z=9)$$

For $\sigma_{\ln c}=0.2$: (a) is 0.30 dex, the draft's "spreading this threshold by 0.3 dex"; (b) is 0.235 dex. A Gaussian fitted directly to the MBK25 selection probability gives $\sigma=0.241,\,0.235,\,0.233$ dex at $z=6,9,12$, i.e. $1.21,\,1.18,\,1.17\,\sigma_{\ln c}$.

So the draft's $1.17$ is correct as the population width, but the reason it gives ("since $M^{1/3}\propto\mu(c)/c^2$") yields only (a) = 1.48. The 1.17 needs the mass dependence of the median concentration ($s_M$). The symbol should be $\sigma(\log M_{\rm vir,DMESF})$, not $\sigma(\log M_{\rm vir,FFB})$.

### R11. Sigmoid width versus concentration scatter: incorrect ($\sigma_{\ln c}=0.18$, "10%")

**Fix:** F3.

$$\text{equal standard deviation:}\quad \sigma_N=\frac{\pi\,\Delta\log M}{\sqrt3}=0.272~{\rm dex}\;\Rightarrow\;\sigma_{\ln c}=\frac{0.272}{1.18}=0.23$$

$$\text{equal central slope:}\quad \frac{1}{4\,\Delta\log M}=\frac{1}{\sigma_N\sqrt{2\pi}}\;\Rightarrow\;\sigma_N=\frac{4\,\Delta\log M}{\sqrt{2\pi}}=0.239~{\rm dex}\;\Rightarrow\;\sigma_{\ln c}=0.20$$

The draft's $0.18=0.272/1.48$ combines the standard-deviation match with coefficient (a), contradicting the 1.17 it quotes. With 1.17 the standard-deviation match gives 0.23 (15% above the adopted 0.2), and the slope match gives 0.20 (2%). The slope match describes the core of the transition; a logistic's standard deviation is inflated by its exponential tails, the same tails that produce the low-redshift FFB population.

The clearest single measure avoids choosing between the two: the range of halo mass over which each selection rises from 16 to 84 per cent.

$$\Delta y_{16\text{-}84}^{\rm sigmoid}=2\,\Delta\log M\,\ln\frac{0.84}{0.16}=3.32\,\Delta\log M=0.50~{\rm dex},\qquad \Delta y_{16\text{-}84}^{\rm MBK25}=2\,\sigma(\log M_t)=2\times0.235=0.47~{\rm dex}$$

The two transitions have nearly the same width. F3 gives replacement text in this form.

## Accelerations in the local Universe

### R12. Milky Way and cluster $g_{\max}$: needs qualification; one claim incorrect

**Fix:** F4; F9 and F10 for the low-redshift population it explains.

$$\frac{g_{\max}}{G}=\frac{g_{\rm vir}}{G}\,\frac{c^2}{2\mu(c)}$$

| halo | definition | $c$ | $R_{\rm vir}$ [kpc] | $g_{\rm vir}/G$ | $g_{\max}/G$ [$M_\odot$ pc$^{-2}$] |
|---|---|---|---|---|---|
| MW $10^{12}\,M_\odot$ | BN98 ($\Delta_c=94$) | 10 | 258 | 15.0 | 505 |
| MW $10^{12}\,M_\odot$ | 200c | 8.71 (median) | 201 | 24.8 | 684 |
| MW $1.5\times10^{12}\,M_\odot$ | BN98 | 10 | 295 | 17.2 | 576 |
| cluster $10^{15}\,M_\odot$ | 200c | 3.84 (median) | 2006 | 248.5 | 2338 |
| cluster $10^{15}\,M_\odot$ | BN98 | 5.3 | 2578 | 150.4 | 2123 |

BK25 quotes $g_{\max}/G\approx500$ for $M_{\rm vir}=10^{12}\,M_\odot$, $c\approx10$ (Bryan & Norman definition). The draft's 576 and 2123 are reproduced only by the inputs in the table, which it does not state. Either quote BK25's 500 with its inputs, or use the model's convention (200c, median $c$): Milky Way 684, cluster 2338.

*Claim to correct.* "No halo in the local Universe attains the critical acceleration through dark matter alone" holds at the median concentration only. With $\sigma_{\ln c}=0.2$ (200c, $z=0$):

| $M_{\rm vir}$ [$M_\odot$] | $10^{14}$ | $3\times10^{14}$ | $10^{15}$ | $2\times10^{15}$ |
|---|---|---|---|---|
| $c$ needed | 8.56 | 6.56 | 4.83 | 4.01 |
| median $\bar c(z=0)$ | 4.80 | 4.22 | 3.84 | 3.82 |
| fraction above $g_{\rm crit}$ | 0.002 | 0.014 | 0.127 | 0.406 |

These equal the MBK25 selection probabilities at $z=0$: high-concentration clusters are the low-redshift MBK25 floor of Figure 8.

### R13. $10^9\,M_\odot$ haloes by $z\approx18$: verified

From R8 at the median concentration: $\log_{10}M_{\rm vir,DMESF}=9.73,\,9.37,\,9.04,\,8.74$ at $z=14,16,18,20$ ($\bar c=3.34$–$3.59$); Li+24 gives $9.71,\,9.37,\,9.07,\,8.80$.

## Star formation in the efficient mode

### R14. Disc radius and dynamical time (draft Eqs. 6–7): verified

Mo, Mao & White (1998): an exponential disc with $j_d/m_d=1$ in a singular isothermal sphere with a flat rotation curve at $V_{\rm vir}$ has $R_d=\lambda R_{\rm vir}/\sqrt2$, so

$$r_{\rm disk}=3R_d=\frac{3}{\sqrt2}\,\lambda\,R_{\rm vir}$$

*Assumptions:* singular isothermal sphere, no adiabatic contraction, specific angular momentum conserved. With $R_{\rm vir}/V_{\rm vir}=\sqrt{2/\Delta}\,H^{-1}$ (from R7):

$$t_{\rm dyn,disk}=\frac{r_{\rm disk}}{V_{\rm vir}}=\frac{3\lambda}{\sqrt\Delta\,H(z)}\propto\lambda\,(1+z)^{-3/2},\qquad \dot m_*=\alpha_{\rm eff}\,m_{\rm ISM}\,\frac{\sqrt\Delta\,H(z)}{3\lambda}$$

This is independent of halo mass. For $\lambda=0.035$ at $z=10$, $t_{\rm dyn,disk}=5.4$ Myr.

### R15. Free-fall time: verified

$$t_{\rm ff}=\sqrt{\frac{3\pi}{32\,G\rho}},\qquad \rho=\mu_m m_p n\;\Rightarrow\;t_{\rm ff}=0.94~{\rm Myr}\left(\frac{\mu_m\,n}{3\times10^3~{\rm cm^{-3}}}\right)^{-1/2}$$

This gives 0.94, 0.83 and 0.79 Myr for $\mu_m=1,\,1.27,\,1.4$ (draft: ~1 Myr).

### R16. Integrated efficiency: needs qualification

**Fix:** F6.

$\epsilon_{\rm SFE}\equiv m_*/(f_bM_{\rm vir})$. $\alpha_{\rm eff}$ is an efficiency per disc dynamical time, so it does not bound $\epsilon_{\rm SFE}$. "Approaching the assumed maximum of $\alpha_{\rm eff}=0.2$" conflates the two.

## Related statements that are not scaling relations

Fixes: F7 (overlap), F8 (Figure 1 caption, "different haloes"), F9 (Figure 8 caption), F10 (§5.3), F11 (number-density figure).

- **Selection overlap (§4).** The draft's numbers (a third at $z=10$, 23% at $z=8$, 63% at $z=12$) match the Millennium-500 runs (33%, 23%, 65%), but they are what independent draws give: the two runs' random-number streams drift apart, so each halo gets an unrelated draw under each criterion. Aligned draws would give 84%, 76% and 95%. The text should say why the overlap is partial (F7).
- **Figure 8 caption.** "Massive enough to cross both thresholds" is incorrect, and so is "insufficient fuel". The selected haloes are groups and clusters below their threshold, drawn from the tails; they keep $\sim10^{10}\,M_\odot$ of cold gas, and the efficient mode raises their star formation rates roughly tenfold (F9, F10).
- **§5.3.** "In rare haloes that still satisfy either criterion despite … declining gas densities": neither criterion uses gas density. Li+24 selects these haloes from the sigmoid tail; MBK25 selects concentrated clusters that genuinely pass $g_{\max}>g_{\rm crit}$.

## Corrections: replacement text for the draft

Each fix gives where the text sits in the draft, quotes the draft, explains what is wrong, and gives replacement LaTeX ready to paste. Numbers are from the Millennium-500 runs (`python plotting/paper_plots.py 98`) unless stated otherwise.

### F1. Figure 1 discussion: thresholds, smoothing and slopes

**Where:** §3.3. **Relations:** R7, R8, R9, R10.

**Draft:**

> The left panel shows the MBK25 acceleration criterion, where halo concentration strongly affects the critical halo mass. The coloured lines mark $g_{\max}=g_{\rm crit}$ at fixed $c$ … with the concentration distribution spreading this threshold by 0.3 dex. … When comparing the two thresholds directly, the FFB threshold scales as $M_{\rm vir,FFB}\propto(1+z)^{-6.2}$ and the MBK25 threshold as $M_{\rm vir,MBK25}\propto(1+z)^{-6.16}$. This common exponent follows from dimensional analysis: at the redshifts of interest the mean halo density tracks the critical density …

**Why:**

The figure now shows the selected galaxies and a single MBK25 line at the median concentration, so the description of coloured lines no longer applies, and the choice of the median concentration needs a reason. The reason is that it is the mass at which half of all haloes are selected: a Gaussian fitted to the MBK25 selection probability is centred within 0.005 dex of it at $z=6$, 9 and 12. The Li+24 threshold is the same point of its sigmoid.

0.3 dex is the shift of one halo's threshold when its concentration moves by $1\sigma$; the spread of the selected population is 0.24 dex (R10). The width is explained in §5.2, so here a pointer is enough.

The exponent $-6.16$ is not recovered: a fit over $6<z<12$ gives $-6.09$ (R9). The dimensional-analysis paragraph repeats a scaling argument made elsewhere, and its premises are loose (R7); it is removed. A corrected version is given below in case it is kept.

**Replacement (§3.3, three paragraphs):**

```latex
\Fig{fig:FFB_selection} shows both selections in the mass--redshift plane. The MBK25 criterion depends on concentration, so it has no single threshold mass; throughout this paper we quote it at the median concentration. This is the mass at which half of all haloes are selected, the same point the Li+24 threshold marks at the midpoint of its sigmoid, so the two thresholds can be compared directly. The left panel shows the MBK25 galaxies and this threshold, $\rm g_{max}=g_{crit}$ at $\rm c\approx3.2$, which gives $\rm M_{vir,MBK25}\approx10^{10.85}\ M_{\odot}$ at $\rm z\approx9$. The galaxies scatter on both sides of the line because of the spread in concentration (\Sec{}).\\
\\
The right panel shows the FFB galaxies and the threshold $\rm M_{vir,FFB}(z)$ of \Eq{}, which at $\rm z\approx9$ is $\rm 10^{10.8}\ M_{\odot}$, 0.045 dex below the MBK25 threshold. Here the spread comes from the sigmoid of \Eq{} (shaded band, $\rm \Delta\log M=0.15$ dex) rather than from a halo property.\\
\\
Both thresholds fall steeply with redshift, $\rm M_{vir,FFB}\propto(1+z)^{-6.2}$ and $\rm M_{vir,MBK25}\propto(1+z)^{-6.1}$ (fitted over $\rm 6<z<12$). They come from different limits on feedback. MBK25 asks whether gravity can hold gas against the momentum young stars inject; FFB asks whether gas turns into stars before that momentum arrives.
```

**Optional, only if the scaling paragraph is kept:**

```latex
When comparing the two thresholds directly, the FFB threshold scales as $\rm M_{vir,FFB}\propto(1+z)^{-6.2}$ and the MBK25 threshold as $\rm M_{vir,MBK25}\propto(1+z)^{-6.1}$ (fitted over $\rm 6<z<12$). The common exponent follows from the halo definition. While matter dominates the expansion, the virial radius scales as $\rm M_{vir}^{1/3}(1+z)^{-1}$ and the virial acceleration as $\rm M_{vir}^{1/3}(1+z)^2$, so a fixed threshold implies $\rm M_{threshold}\propto(1+z)^{-6}$. The MBK25 exponent departs from this only because the median concentration rises with redshift at fixed mass.
```

### F2. DMESF/MBK25 exponent in the Conclusions

**Where:** Conclusions. **Relations:** R9.

**Draft:**

> … scaling $M_{\rm vir,MBK25}\propto(1+z)^{-6.16}$ and $M_{\rm vir,FFB}\propto(1+z)^{-6.2}$ arising from different physical motivations.

**Why:**

As F1: a fit over $6<z<12$ gives $-6.09$, and the value moves between $-6.06$ and $-6.21$ with the fitting range, so only $-6.1$ is meaningful.

**Replacement:**

```latex
... scaling $\rm M_{vir,MBK25}\propto(1+z)^{-6.1}$ and $\rm M_{vir,FFB}\propto(1+z)^{-6.2}$ arising from different physical motivations.
```

### F3. Sigmoid width and concentration scatter

**Where:** §5.2. **Relations:** R10, R11.

**Draft:**

> Since $M_{\rm vir,MBK25}^{1/3}\propto\mu(c)/c^2$, a log-normal scatter in concentration of width $\sigma_{\ln c}$ induces a scatter in threshold mass of $\sigma(\log M_{\rm vir,FFB})\approx1.17\,\sigma_{\ln c}$. A logistic transition of width $\Delta\log M$ has standard deviation $\pi\Delta\log M/\sqrt3$, so the $\Delta\log M=0.15$ adopted by Li+24 corresponds to $\sigma_{\ln c}=0.18$ … with the two agreeing to 10%.

**Why:**

$M^{1/3}\propto\mu/c^2$ alone gives a coefficient of 1.48, not 1.17; the 1.17–1.18 needs the median concentration's rise with mass as well. The 0.18 then contradicts the 1.17 ($0.272/1.17=0.23$), and the 10% follows from that error. The subscript should be MBK25.

The clearest comparison is the mass range over which each selection rises from 16 to 84 per cent: 0.47 dex for the concentration scatter and 0.50 dex for the sigmoid (R11). Far from the threshold the sigmoid falls off more slowly. The phrase "non-integrated galaxy properties such as the stellar mass function" is also changed: the stellar mass function is a population statistic.

**Replacement (§5.2, both paragraphs):**

```latex
The Li+24 framework uses a sigmoid to smooth the transition from non-FFB to FFB galaxies, so that a sharp threshold does not put an artificial feature into population statistics such as the stellar mass function. The MBK25 model gives this smoothing a physical origin: at fixed halo mass, haloes span a log-normal range of concentrations, and more concentrated haloes cross the threshold for efficient star formation at lower mass.\\
\\
Since $\rm M_{vir,MBK25}^{1/3}\propto\mu(c)/c^2$, and the median concentration rises weakly with mass, a log-normal scatter in concentration of $\rm \sigma_{ln\ c}=0.2$ spreads the threshold mass by 0.24 dex: the fraction of haloes selected rises from 16 to 84 per cent over 0.47 dex in halo mass. The Li+24 sigmoid with $\rm \Delta\log M=0.15$ rises over the same range in 0.50 dex. The smoothing that Li+24 introduces by hand therefore matches the concentration scatter already present in the model. The two differ only far from the threshold, where the sigmoid falls off more slowly.
```

### F4. Accelerations in the local Universe

**Where:** §3.2. **Relations:** R12.

**Draft:**

> The Milky Way's halo reaches only $g_{\max}/G\approx576\,M_\odot\,{\rm pc}^{-2}$ and a cluster-mass halo $\approx2123$, both below $g_{\rm crit}$, no halo in the local Universe attains the critical acceleration through dark matter alone.

**Why:**

The halo masses, concentrations and halo definition behind 576 and 2123 are not stated and are not the model's conventions (576 needs $1.5\times10^{12}\,M_\odot$, $c=10$ and a Bryan & Norman radius). With the model's 200c definition and median concentrations the values are 684 and 2338.

The last clause is wrong under the model's own concentration scatter: 13% of $10^{15}\,M_\odot$ clusters at $z=0$ exceed $g_{\rm crit}$. These are the low-redshift MBK25 galaxies of Figure 8.

**Replacement:**

```latex
At $\rm z=0$ a Milky Way-mass halo ($\rm 10^{12}\ M_{\odot}$, median $\rm c=8.7$) reaches only $\rm g_{max}/G\approx680\ M_{\odot}\ pc^{-2}$, and a $\rm 10^{15}\ M_{\odot}$ cluster (median $\rm c=3.8$) $\rm \approx2300\ M_{\odot}\ pc^{-2}$, both below $\rm g_{crit}$. Through dark matter alone, only the most concentrated clusters reach it: about 13 per cent of $\rm 10^{15}\ M_{\odot}$ haloes, those more than $1.1\sigma$ above the median concentration.
```

### F5. Li+24 redshift floor

**Where:** §5.3. **Relations:** R3.

**Draft:**

> The Li+24 analytic framework imposes a quenching mass of $10^{12}\,M_\odot$ and a redshift floor of $z>6$ (which comes automatically from the $M_{\rm vir,FFB}$ variation with redshift and the quenching condition) on the FFB population …

**Why:**

Equation 1 reaches $10^{12}\,M_\odot$ at $z=5.4$, not 6.

**Replacement:**

```latex
The Li+24 analytic framework imposes a quenching mass of $\rm 10^{12}\ M_{\odot}$ on the FFB population. Because $\rm M_{vir,FFB}$ rises towards lower redshift (\Eq{}), the threshold reaches the quenching mass at $\rm z\approx5.4$, which sets an effective redshift floor ...
```

### F6. Integrated versus per-dynamical-time efficiency

**Where:** §4.2. **Relations:** R16.

**Draft:**

> Highly efficient star-forming galaxies … show $\epsilon_{\rm SFE}\sim0.01-0.2$, approaching the assumed maximum of $\alpha_{\rm eff}=0.2$.

**Why:**

$\alpha_{\rm eff}$ is the fraction of the ISM converted per disc dynamical time; $\epsilon_{\rm SFE}=m_*/(f_bM_{\rm vir})$ integrates over the halo's history. One does not bound the other.

**Replacement:**

```latex
... show $\rm \epsilon_{SFE}\sim0.01$--$0.2$, comparable to the per-dynamical-time efficiency $\rm \alpha_{eff}=0.2$. The two are different quantities: $\rm \alpha_{eff}$ sets the fraction of the interstellar medium converted per disc dynamical time, while $\rm \epsilon_{SFE}$ integrates over the halo's history, so $\rm \alpha_{eff}$ does not cap it.
```

### F7. Selection overlap

**Where:** §4. **Relations:** R9.

**Draft:**

> The two criteria do select different haloes. Using the same galaxies across the two runs, a third selected by either criterion at $z=10$ is selected by both, falling to 23% at $z=8$ and then rising to 63% at $z=12$ … every median in Figure 5 differs by at most 0.047 dex, with an RMS difference of 0.039 dex, over $6<z<12$. The agreement is therefore a property of the threshold curves rather than of the individual selections.

**Why:**

The overlap numbers are right for the current Millennium-500 runs (33% at $z=10.1$, 23% at $z=8.6$, 65% at $z=11.9$), but the reason is not stated. Each criterion is a random draw against its selection probability, and in these runs the two draws are independent: the measured overlap equals the overlap the two probability curves predict for independent draws at every redshift (0.33, 0.24 and 0.64). In the 62.5 Mpc/$h$ test box the draws happened to stay aligned and the overlap was 77%; in the large box they are not, because each run's random-number stream drifts out of step once the two models' galaxy populations differ.

Were each halo given the same draw under both criteria, the overlap would be 84%, 76% and 95% at the same redshifts. A halo-keyed draw that does this is implemented in `src/model_regimes.c` on `dev`; after a rerun, replace this paragraph with the aligned numbers.

0.047 and 0.039 dex are differences between the two threshold curves, not between medians in Figure 5. The replacement quotes the Figure 5 medians instead (SFR 0.75 and 0.70; metallicity 0.042 and 0.041 $Z_\odot$).

**Replacement (§4, current runs):**

```latex
Both criteria are applied as random draws against their selection probabilities, redrawn at every snapshot, and in our runs the draws for the two criteria are independent. Matching haloes by position across the two runs, 33 per cent of those selected by either criterion at $\rm z=10$ are selected by both, 23 per cent at $\rm z\approx8.5$ and 65 per cent at $\rm z\approx12$, where most haloes lie above both thresholds. These are the overlaps the two selection probabilities predict for independent draws; had each halo been given the same draw under both criteria, they would be 84, 76 and 95 per cent. The medians in \Fig{fig:ffb_props} agree to 0.05 dex in star formation rate and 0.01 dex in metallicity, so the population statistics do not depend on which individual haloes are drawn.
```

### F8. Figure 1 caption and "different haloes" in §5.4

**Where:** Figure 1 caption, §5.4. **Relations:** R9.

**Draft:**

> (Figure 1) … Left is the MBK25 model's maximum acceleration criterion … shown for different concentrations (coloured lines), and the points are the selected galaxies coloured by concentration. Right is the FFB model's threshold mass, with the shaded region indicating the width of the sigmoid function. (Earlier draft: … while selecting different individual haloes within it.) (§5.4) The two different models do select different haloes while producing the same overall population, so the correspondence we report holds at the population level.

**Why:**

The left panel now has a single line at the median concentration and grey points, so the coloured lines and points no longer exist. Galaxies below that line need explaining: they sit in haloes more concentrated than the median. The shaded band spans selection probabilities of 0.1 to 0.9, and each panel is a random subsample of 7,500 of the galaxies selected at $4<z<16$, so point density is not the true number of selected galaxies. The criteria share only part of their individual haloes in the current runs (F7), so the caption no longer claims either way.

**Replacement (Figure 1 caption):**

```latex
Halo selection by the two criteria in the mass--redshift plane. \textit{Left}: galaxies selected by MBK25 (grey points) and the threshold $\rm g_{max}(M_{vir},z,c)=g_{crit}$ at the median concentration, $\rm c\approx3.2$ (purple line). Galaxies below the line sit in haloes more concentrated than the median. \textit{Right}: galaxies selected by FFB (grey points) and the threshold mass $\rm M_{vir,FFB}(z)$ (red line), with the shaded band spanning selection probabilities of 0.1 to 0.9 in the sigmoid. Points in both panels are a random subsample of all galaxies selected at $\rm 4<z<16$.
```

**Replacement (§5.4):**

```latex
The two models select the same region of the mass--redshift plane and produce the same overall population, while sharing only part of their individual haloes (\Sec{}).
```

### F9. Figure 8 caption: low-redshift population

**Where:** Figure 8 caption. **Relations:** R12.

**Draft:**

> Both models produce a small fraction of efficient galaxies at lower redshifts ($\approx10^{-6}$ at $z=1$); these haloes are massive enough to cross both thresholds and are often massive elliptical galaxies with insufficient fuel for star formation.

**Why:**

The fraction at $z=1$ is $1.4\times10^{-5}$ for MBK25 and $2.4\times10^{-7}$ for FFB, not $10^{-6}$. MBK25 continues to $z=0$ (30–80 galaxies per snapshot, a fraction of $2.8\times10^{-6}$); FFB has none below $z=0.56$.

These haloes do not cross the thresholds. They are groups and clusters ($M_{\rm vir}\approx10^{13}$–$10^{15.2}\,M_\odot$, median 0.2–0.8 dex below their run's threshold) selected from the tails: Li+24 from the sigmoid, MBK25 from concentrations well above the median (F4).

They are not short of fuel. They hold $\sim10^{10}\,M_\odot$ of cold gas, and the efficient mode raises their star formation rates by an order of magnitude: at $z<1$ the median is 40–150 $M_\odot$ yr$^{-1}$, against 3–15 $M_\odot$ yr$^{-1}$ for the same haloes in the run without either mode.

**Replacement:**

```latex
Both models produce a small fraction of efficient galaxies at lower redshifts: at $\rm z=1$ the fraction is $\rm 1.4\times10^{-5}$ for MBK25 and $\rm 2\times10^{-7}$ for FFB, and MBK25 continues to $\rm z=0$ ($\rm 3\times10^{-6}$). These are groups and clusters ($\rm M_{vir}\approx10^{13}$--$\rm 10^{15}\ M_{\odot}$) selected from the tails of the two selection functions: haloes below $\rm M_{vir,FFB}$ drawn from the Li+24 sigmoid, and clusters concentrated enough to reach $\rm g_{crit}$ at their centres.
```

### F10. Origin and effect of the low-redshift population

**Where:** §5.3. **Relations:** R8, R12.

**Draft:**

> A secondary consequence is a sparse population of FFB and MBK25 galaxies that persists to lower redshifts, seen in Figure 8, in rare haloes that still satisfy either implementation's criterion despite the generally declining gas densities from cosmic noon to today.

**Why:**

Neither criterion uses gas density. The Li+24 threshold is a power law that reaches $10^{17}\,M_\odot$ by $z=0$, far above any halo, so its tail ends at $z\approx0.6$. The MBK25 threshold stops falling once dark energy dominates, flattening to $\approx10^{15.4}\,M_\odot$ at $z=0$, so concentrated clusters still pass it.

The population is small but not harmless. Its galaxies are cluster and group centrals with $\sim10^{10}\,M_\odot$ of cold gas, and the efficient mode raises their star formation rates roughly tenfold (40–150 against 3–15 $M_\odot$ yr$^{-1}$ at $z<1$). Individual galaxies are selected for 1.6 (FFB) and 2.0 (MBK25) snapshots on average, but the most massive, concentrated clusters are reselected for up to 40 snapshots.

**Replacement:**

```latex
A secondary consequence is a sparse population of FFB and MBK25 galaxies at lower redshifts (\Fig{}). Below $\rm z\approx4$ the two fractions diverge. The Li+24 threshold is the power law of \Eq{}, which exceeds $\rm 10^{17}\ M_{\odot}$ by $\rm z=0$, so its tail ends at $\rm z\approx0.6$. The MBK25 threshold stops falling once dark energy dominates the expansion; it flattens to $\rm \approx10^{15.4}\ M_{\odot}$ at $\rm z=0$, and clusters with concentrations 1--3$\sigma$ above the median still exceed it, giving a fraction of $\rm 3\times10^{-6}$ at $\rm z=0$. These are group and cluster centrals that still hold $\rm \sim10^{10}\ M_{\odot}$ of cold gas, and the efficient mode raises their star formation rates roughly tenfold, to 40--150 $\rm M_{\odot}\ yr^{-1}$ at $\rm z<1$. Both criteria are re-evaluated at every snapshot, so most are selected for one or two snapshots, although the most massive, concentrated clusters are reselected repeatedly. Neither criterion was constructed for $\rm z\lesssim4$, and this population is not a prediction of feedback-free starbursts.
```

### F11. Number-density figure: method and caption

**Where:** "Comparing the two criteria" and its figure (`G_mstar_vs_z_ndensity.pdf`). **Relations:** —.

**Draft:**

> To determine whether the FFB and MBK25 implementations are acting as intended, we compare, at fixed cumulative comoving number density, the abundance of galaxies probed by the deep JWST fields in Figure X.

**Why:**

The text does not say how the curves are made or why the result is a stellar mass against redshift. The figure tests the models against observations rather than whether the implementations act as intended. The caption's observation labels should name the source papers (Baggen+23 diamonds, Casey+24 squares, Sun+24 triangles), and the density plotted ($10^{-5}$ Mpc$^{-3}$) should be checked against the run's printed output.

**Replacement (text):**

```latex
JWST surveys find the most massive galaxies in their volume, so a median over a model population is not the right comparison. Instead we compare at fixed cumulative comoving number density. At each snapshot we rank every resolved galaxy (at least 20 particles, centrals and satellites, efficient or not) by stellar mass and take the stellar mass of the $\rm (nV)$-th most massive, where $\rm V$ is the simulated comoving volume. This is the stellar mass above which galaxies have a number density $\rm n$. Repeating this at every snapshot from $\rm z=4$ to 15 gives the stellar mass of galaxies of that rarity as a function of redshift, shown in \Fig{fig:ffb_dist} for $\rm n=10^{-5}\ Mpc^{-3}$. This is roughly one galaxy per volume of the deep JWST fields, so an observed galaxy is reproduced when the curve passes through it. The shaded bands are the $1\sigma$ scatter between the eight octants of the box.
```

**Replacement (caption, first sentences):**

```latex
Stellar mass of galaxies at a fixed cumulative comoving number density of $\rm n=10^{-5}\ Mpc^{-3}$, as a function of redshift, for the MBK25 implementation (purple), the FFB model (black) and a model in which neither mode operates (red). At each snapshot this is the stellar mass of the $\rm (nV)$-th most massive galaxy in the simulated volume $\rm V$; bands show the scatter between the eight octants of the box.
```
