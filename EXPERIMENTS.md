### 1. Text vs Sound Targets
* Matches brain scans to text and sound vectors. Does the model read actual words, or just hear host voice pitch and room echo? Is this a mind reader or a sound meter?
* Train one decoder to guess text vectors, and another to guess sound vectors (`LCO-Embedding-Omni-7B`) from the exact same brain scans.

---

### 2. Cross-Subject Test
* All brain maps share one surface space. If trained on Person 0 and tested on Person 1, do the brains match? Are we all just copies running on different bone hardware?
* Train the decoder using brain data from Person A, then test if it can decode Person B using the shared brain surface map (`fsaverage5`).

---

### 3. Whole-Brain vs Local Parts
* The baseline uses all surface points. Do foot movement zones help decode a podcast? Block out everything except hearing and language zones... can the language zone carry the whole load?
* Turn off most of the 20,484 brain surface points, keeping only hearing and language zones, then compare that score against the whole brain.

---

### 4. Hard and Easy Clip Checks
* Some podcast parts match fast; others fail. Why? Does the model struggle with long words, or get lost during ad breaks? Track the bad clips.
* Sort all podcast clips by decoder score and analyze shared traits among failing clips (e.g., word length, unusual sentence structure, ad breaks).

---

### 5. Time Lag Checks
* Blood moves slow, thought moves fast. Waiting 2.5 seconds for blood flow is standard... what about 3 seconds or 1 second? A clock off by a hair reads a brain thinking about lunch, not the story.
* Test different delay windows (1s to 5s) instead of relying solely on the default 2.5-second blood-flow delay to find the optimal timing.

---

### 6. Shape Check Before Training
* Why build big neural networks before checking if shapes fit? Forcing a square block into a round hole with a giant hammer isn't optimal. Check raw data shapes first.
* Measure alignment between raw brain shapes and text/sound shapes using basic mathematical distance checks before model training.

---

### 7. Retrieval vs Text Writing
* Picking the right clip from a list of 100 is easy; writing a sentence from scratch is hard. If the model picks the right clip, can it actually generate text, or just wild noise?
* Take a model that excels at picking the right podcast clip from a 100-item lineup, and test its ability to generate brand-new sentences from scratch.

---

### 8. Word Repeat Checks
* Hosts say "welcome back" fifty times. The model learns catchphrases, not the podcast... like a student gaming the test. Strip out repeat words.
* Group podcast samples by word/phrase frequency to verify whether high decoder scores only occur on common phrases.
