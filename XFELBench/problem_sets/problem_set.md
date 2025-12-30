QA-1 (European XFEL – Detectors)
Question
Why can the AGIPD (Adaptive Gain Integrating Pixel Detector) operate at MHz repetition rates at the European XFEL, and what are the main design trade-offs?
Answer
AGIPD is specifically designed for the European XFEL pulse structure, which consists of pulse trains at 10 Hz with intra-train repetition rates up to several MHz. The detector uses in-pixel analog memory to store multiple images during a pulse train, enabling data acquisition at MHz rates without immediate readout. After each train, data are read out during the inter-train gap.
To achieve both single-photon sensitivity and very large dynamic range, AGIPD employs adaptive gain switching, automatically adjusting pixel gain depending on signal strength.
The main trade-offs include increased pixel complexity, limited pixel fill factor, higher power consumption, and more complex calibration procedures due to multi-gain operation and radiation hardness requirements.

QA-2 (European XFEL – Single Particle Imaging)
Question
In MHz single-particle imaging (SPI) experiments at the European XFEL, why are detector noise and background stability as critical as high repetition rate?
Answer
SPI relies on collecting and classifying a very large number of weak diffraction patterns, often close to the single-photon limit. Detector noise directly affects the ability to distinguish true photon hits from electronic background, influencing hit finding, orientation recovery, and final reconstruction quality.
Background instability introduces systematic errors in background subtraction, which propagate through classification and averaging steps. At MHz repetition rates, even small instabilities are amplified by the enormous data volume, leading to biased reconstructions or reduced achievable resolution.
QA-3 (Arrival Time Monitoring)
Question
How does an arrival time monitor convert femtosecond timing jitter between XFEL and optical laser pulses into a measurable signal?
Answer
Arrival time monitors typically exploit XFEL-induced ultrafast changes in a material’s optical properties. A small fraction of the XFEL pulse excites the material, while a synchronized optical probe pulse samples the induced transient response.
By using temporal-to-spatial or temporal-to-spectral encoding (for example, with a chirped optical pulse), the relative arrival time between XFEL and optical laser pulses is mapped onto a measurable spatial position or wavelength shift on a camera. This allows shot-by-shot timing information to be recorded and used to correct pump–probe data.

QA-4 (In-situ Timing Diagnostics)
Question
What is the principle of a self-referenced in-situ arrival time monitor, and what systematic errors does it reduce compared to external diagnostics?
Answer
A self-referenced in-situ arrival time monitor measures the relative delay between XFEL and optical laser pulses directly at or very near the experimental interaction point. By using common optical paths or dual reference pulses with a fixed internal delay, it minimizes sensitivity to slow drifts such as thermal or mechanical instabilities.
Compared to diagnostics placed on separate beamlines, this approach reduces systematic errors arising from non-common beam paths and ensures that the measured timing corresponds more accurately to the actual pump–probe delay experienced by the sample.

QA-5 (LCLS / LCLS-II – Machine Learning)
Question
Why are machine learning methods particularly effective for accelerator tuning and beamline optimization at LCLS and LCLS-II?
Answer
XFEL accelerators are high-dimensional, nonlinear systems with strong parameter coupling and noisy diagnostics. Machine learning models, such as surrogate models combined with Bayesian optimization or reinforcement learning, can efficiently explore parameter space with far fewer measurements than manual scans.
These approaches enable faster tuning, improved stability, adaptive compensation of drifts, and automated optimization under operational constraints, which is especially valuable at high repetition rates and during user operations.

QA-6 (LCLS-II – High Repetition Rate Challenges)
Question
What new data acquisition and analysis challenges are introduced by LCLS-II compared to the original LCLS?
Answer
LCLS-II increases the repetition rate from up to 120 Hz to as high as 1 MHz, leading to orders-of-magnitude higher data rates. This creates challenges in real-time data reduction, storage bandwidth, and offline analysis scalability.
To address these challenges, LCLS-II relies on streaming architectures, on-the-fly data filtering, heterogeneous computing (CPU/GPU), and closer integration between detectors, data acquisition systems, and analysis pipelines.

QA-7 (European XFEL – Pulse Train Structure)
Question
How does the pulse train structure of the European XFEL influence detector and experiment design?
Answer
The European XFEL delivers bursts of up to thousands of pulses at MHz repetition rates, followed by long gaps between pulse trains. Detectors must therefore support burst-mode acquisition with local storage, while experiments must tolerate non-uniform temporal sampling.
This structure strongly affects detector electronics, synchronization systems, and data analysis strategies, requiring train-aware calibration, buffering, and metadata handling.

QA-8 (Hard X-ray Stability)
Question
Why is beam stability particularly important for hard X-ray experiments?
Answer
Hard X-ray experiments often probe subtle structural changes or weak scattering signals. Fluctuations in photon energy, pulse energy, beam position, or wavefront can significantly degrade data quality.
At Hard X-ray FEL, maintaining high stability is essential for reproducible pump–probe measurements, high-resolution diffraction, and precision spectroscopy, especially when operating near damage thresholds or at low signal levels.

QA-9 (Timing Jitter Correction)
Question
Why is shot-by-shot timing correction preferred over statistical averaging in XFEL pump–probe experiments?
Answer
XFEL timing jitter between X-ray and optical pulses can be comparable to or larger than the timescales of ultrafast dynamics under study. Statistical averaging without correction effectively blurs temporal resolution.
Shot-by-shot timing correction allows experimental data to be re-binned according to the true pump–probe delay, restoring femtosecond or even sub-femtosecond temporal resolution.

1. Relative arrival time of X-ray and optical laser can be measured by spectral and spatial encoding schemes at low repetiition rate XFEL, but what are the challenges for the measurement at high repetition XFEL sources? How does European XEFL or LCLS address this problem? 
2. how to resolve the Resonant Auger scattering of a highly distorted molecule in time?
3. Is it possible to use seed laser and seeded FEL for pump to solve the temporal jitter problem in case of pump-probe experiments, although the pulse durations of seed laser and seeded FEL are significantly longer than expected?
4. There are three primary sample delivery systems in Serial crystallography: fixed-targets, liquid injection, and hybrid methods. Please give a comparison of these three systems in detail, especially including their advantatages and limitations. 
5. What is wavefront sensing and why it is important for XFEL? How many techniques are used for wavefront sensing at LCLS? What are the advantages and disadvantages for those techniques? Are there any trade-offs between them?新增一篇文献，上述有1~2个