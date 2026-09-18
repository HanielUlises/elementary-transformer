import ElementaryTransformer.Semantics

/-!
# Distinguishing certificates

A certificate claims that a sentence of FOᵏ with quantifier rank at most `q`
holds in one finite structure and fails in another. `Certificate.check` decides
the claim and `Certificate.check_sound` proves that an accepted certificate is
correct with respect to the Tarskian semantics.
-/

namespace ElementaryTransformer

structure Certificate where
  left : FinStructure
  right : FinStructure
  formula : Formula
  k : Nat
  q : Nat

/-- The assignment used to evaluate sentences; any other gives the same result. -/
def ρ₀ : Nat → Nat := fun _ => 0

namespace Certificate

def check (c : Certificate) : Bool :=
  c.formula.isSentence && c.formula.usesBelow c.k && decide (c.formula.rank ≤ c.q) &&
    c.formula.eval c.left ρ₀ && !c.formula.eval c.right ρ₀

theorem check_sound (c : Certificate) (h : c.check = true) :
    c.formula.isSentence = true ∧ c.formula.usesBelow c.k = true ∧ c.formula.rank ≤ c.q ∧
      (∀ ρ, c.formula.Sat c.left ρ) ∧ (∀ ρ, ¬ c.formula.Sat c.right ρ) := by
  simp only [check, Bool.and_eq_true, decide_eq_true_eq, Bool.not_eq_true'] at h
  obtain ⟨⟨⟨⟨hs, hk⟩, hq⟩, hl⟩, hr⟩ := h
  refine ⟨hs, hk, hq, fun ρ => ?_, fun ρ hsat => ?_⟩
  · exact (Formula.sentence_sat_iff _ _ hs ρ₀ ρ).mp ((Formula.eval_iff_sat _ _ _).mp hl)
  · have := (Formula.eval_iff_sat _ _ _).mpr ((Formula.sentence_sat_iff _ _ hs ρ ρ₀).mp hsat)
    rw [hr] at this
    exact Bool.false_ne_true this

end Certificate

end ElementaryTransformer
