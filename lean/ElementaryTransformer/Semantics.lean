import ElementaryTransformer.Syntax

/-!
# Semantics and a verified model checker

A finite structure has universe `{0, …, size - 1}` and interprets every
relation symbol by a decidable predicate on lists of elements. `Formula.Sat`
is the Tarskian satisfaction relation, with quantifiers ranging over the
universe; `Formula.eval` is the obvious evaluator. `eval_iff_sat` proves that
they agree, and `sat_congr` shows that satisfaction only depends on the values
of the free variables, so that a sentence has the same truth value under every
assignment.
-/

namespace ElementaryTransformer

structure FinStructure where
  size : Nat
  rel : String → List Nat → Bool

/-- The assignment `ρ` with variable `i` sent to `a`. -/
def update (ρ : Nat → Nat) (i a : Nat) : Nat → Nat :=
  fun j => if j = i then a else ρ j

namespace Formula

def Sat (M : FinStructure) : (Nat → Nat) → Formula → Prop
  | _, tt => True
  | _, ff => False
  | ρ, rel R args => M.rel R (args.map ρ) = true
  | ρ, eq i j => ρ i = ρ j
  | ρ, neg φ => ¬ Sat M ρ φ
  | ρ, conj φ ψ => Sat M ρ φ ∧ Sat M ρ ψ
  | ρ, disj φ ψ => Sat M ρ φ ∨ Sat M ρ ψ
  | ρ, ex i φ => ∃ a, a < M.size ∧ Sat M (update ρ i a) φ
  | ρ, all i φ => ∀ a, a < M.size → Sat M (update ρ i a) φ

def eval (M : FinStructure) : (Nat → Nat) → Formula → Bool
  | _, tt => true
  | _, ff => false
  | ρ, rel R args => M.rel R (args.map ρ)
  | ρ, eq i j => ρ i == ρ j
  | ρ, neg φ => !eval M ρ φ
  | ρ, conj φ ψ => eval M ρ φ && eval M ρ ψ
  | ρ, disj φ ψ => eval M ρ φ || eval M ρ ψ
  | ρ, ex i φ => (List.range M.size).any fun a => eval M (update ρ i a) φ
  | ρ, all i φ => (List.range M.size).all fun a => eval M (update ρ i a) φ

theorem eval_iff_sat (M : FinStructure) (φ : Formula) (ρ : Nat → Nat) :
    eval M ρ φ = true ↔ Sat M ρ φ := by
  induction φ generalizing ρ with
  | tt => simp [eval, Sat]
  | ff => simp [eval, Sat]
  | rel R args => simp [eval, Sat]
  | eq i j => simp [eval, Sat]
  | neg φ ih => simp [eval, Sat, ← ih]
  | conj φ ψ ihφ ihψ => simp [eval, Sat, ihφ, ihψ]
  | disj φ ψ ihφ ihψ => simp [eval, Sat, ihφ, ihψ]
  | ex i φ ih => simp [eval, Sat, List.any_eq_true, List.mem_range, ih]
  | all i φ ih => simp [eval, Sat, List.all_eq_true, List.mem_range, ih]

instance (M : FinStructure) (ρ : Nat → Nat) (φ : Formula) : Decidable (Sat M ρ φ) :=
  decidable_of_iff _ (eval_iff_sat M φ ρ)

theorem sat_congr (M : FinStructure) (φ : Formula) :
    ∀ (bound : List Nat) (ρ σ : Nat → Nat), closedUnder bound φ = true →
      (∀ i, i ∈ bound → ρ i = σ i) → (Sat M ρ φ ↔ Sat M σ φ) := by
  induction φ with
  | tt => intros; simp [Sat]
  | ff => intros; simp [Sat]
  | rel R args =>
    intro bound ρ σ hc hb
    simp only [closedUnder, List.all_eq_true, List.contains_iff_mem] at hc
    have : args.map ρ = args.map σ := List.map_congr_left fun a ha => hb a (hc a ha)
    simp [Sat, this]
  | eq i j =>
    intro bound ρ σ hc hb
    simp only [closedUnder, Bool.and_eq_true, List.contains_iff_mem] at hc
    simp [Sat, hb i hc.1, hb j hc.2]
  | neg φ ih =>
    intro bound ρ σ hc hb
    simp only [Sat, ih bound ρ σ hc hb]
  | conj φ ψ ihφ ihψ =>
    intro bound ρ σ hc hb
    simp only [closedUnder, Bool.and_eq_true] at hc
    simp only [Sat, ihφ bound ρ σ hc.1 hb, ihψ bound ρ σ hc.2 hb]
  | disj φ ψ ihφ ihψ =>
    intro bound ρ σ hc hb
    simp only [closedUnder, Bool.and_eq_true] at hc
    simp only [Sat, ihφ bound ρ σ hc.1 hb, ihψ bound ρ σ hc.2 hb]
  | ex i φ ih =>
    intro bound ρ σ hc hb
    have step : ∀ a, Sat M (update ρ i a) φ ↔ Sat M (update σ i a) φ := fun a =>
      ih (i :: bound) _ _ hc fun j hj => by
        by_cases h : j = i
        · simp [update, h]
        · simp only [update, h, if_false]
          exact hb j (by simpa [h] using hj)
    simp only [Sat, step]
  | all i φ ih =>
    intro bound ρ σ hc hb
    have step : ∀ a, Sat M (update ρ i a) φ ↔ Sat M (update σ i a) φ := fun a =>
      ih (i :: bound) _ _ hc fun j hj => by
        by_cases h : j = i
        · simp [update, h]
        · simp only [update, h, if_false]
          exact hb j (by simpa [h] using hj)
    simp only [Sat, step]

/-- A sentence has the same truth value under every assignment. -/
theorem sentence_sat_iff (M : FinStructure) (φ : Formula) (h : isSentence φ = true)
    (ρ σ : Nat → Nat) : Sat M ρ φ ↔ Sat M σ φ :=
  sat_congr M φ [] ρ σ h (by simp)

end Formula

end ElementaryTransformer
