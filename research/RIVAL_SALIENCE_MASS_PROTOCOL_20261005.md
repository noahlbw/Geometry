# Surviving observer salience mass

Same64 developed top-left512 windows with original full-image wide context.
Freeze Geometry, finite VIP wide, original fine views, all20 vocabulary, hard
eligibility, canonical/self/invalid protection, projection/posterior solver and H.
No additional observation, threshold, temperature, parameter or domain routing.

The count/fixed-mass ablation showed count compensation can help or hurt by
domain. Test replacing only the compensation term, preserving actual per-alias
hard counterfactual evidence and its competitor-specific decisions.

Original D_count contains log(K/remaining)/beta. New D_prior replaces it with
-log(sum_kept pi_alias)/beta in every original wide crop before coverage averaging.
Wide candidate pi=softmax(original wide crop salience). Primary fine candidate pi
mixes original fine crop salience distributions under original query coverage,
then normalizes each query/class. Log-space evaluation avoids underflow. All risk
decisions remain unchanged. Uniform pi recovers the old count rule; zero risk is
an exact identity. Salience already enters the original profiled logits and is
not moved or reprofiled. These are observer priors, NOT correctness probabilities
or a new posterior model of all20 scores.

Candidates: FineSalienceMass_Projected (primary), WideSalienceMass_Projected.
Each has its own canonical-preserving noncanonical class mean, an equal-strength
previous hard direction, and three matched prior-alias permutations. Permutations
hold every query/class prior spectrum and risk decision fixed, but deliberately
change surviving prior mass; they do NOT match final normalization budgets.
Class mean preserves canonical prior mass and total class mass, not each rival's
surviving mass. Equal-strength control uses one alias-derived window scalar.
Prior permutations test the new normalization information, not the correctness of
the earlier binary selector. Keep fixed-mass hard as an explicitly unsafe mean-
high diagnostic, never automatically promote it as a new screening contribution.

Persist score/label-free diagnostics before masks. Verify five frozen scores and
per-image endpoints,64 unique keys, unchanged identities/weights, confusion and
target sums, transition endpoints, canonical0, normalized priors, uniform/count
identity, own permutation spectra and class-mean mass. Independently time warm
hard/two candidates in synchronized alternating three repetitions with resident
memory and real singleton/all-arm equality checks. Candidate timing includes
small CPU score checks; no sub-percent speed claim. Singleton writers do no extra
diagnostic controls or unused candidate computation.

Frozen accuracy gate: mean above previous projected hard, at least five/eight main
domain wins (LoveDA D once), and no reported protocol loss>1pp (including LoveDA P).
Mechanism gate additionally requires each candidate's mean above its own mean-
prior, equal-strength and three-seed alias-null mean. Fixed-mass diagnostic mean
is reported but does not define this gate because it already violates the stability
criterion on VDD/LoveDA P. A passing candidate still needs complete-image testing.
No automatic full20092 rollout, retained-model replacement or diagnostic promotion.
