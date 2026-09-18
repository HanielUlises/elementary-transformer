/-!
# Syntax of first-order logic over relational signatures

Variables are natural numbers and relation symbols are named by strings. A
formula belongs to the `k`-variable fragment FOᵏ when all its variables, free
or bound, are smaller than `k`; variables may be requantified.
-/

namespace ElementaryTransformer

inductive Formula where
  | tt
  | ff
  | rel (sym : String) (args : List Nat)
  | eq (i j : Nat)
  | neg (φ : Formula)
  | conj (φ ψ : Formula)
  | disj (φ ψ : Formula)
  | ex (i : Nat) (φ : Formula)
  | all (i : Nat) (φ : Formula)
  deriving Repr, BEq, Inhabited

namespace Formula

/-- Quantifier rank: the maximal nesting depth of quantifiers. -/
def rank : Formula → Nat
  | tt | ff | rel .. | eq .. => 0
  | neg φ => rank φ
  | conj φ ψ | disj φ ψ => max (rank φ) (rank ψ)
  | ex _ φ | all _ φ => rank φ + 1

/-- `usesBelow k φ` holds iff every variable occurring in `φ` is smaller than `k`. -/
def usesBelow (k : Nat) : Formula → Bool
  | tt | ff => true
  | rel _ args => args.all (· < k)
  | eq i j => decide (i < k) && decide (j < k)
  | neg φ => usesBelow k φ
  | conj φ ψ | disj φ ψ => usesBelow k φ && usesBelow k ψ
  | ex i φ | all i φ => decide (i < k) && usesBelow k φ

/-- `closedUnder bound φ` holds iff every free variable of `φ` belongs to `bound`. -/
def closedUnder (bound : List Nat) : Formula → Bool
  | tt | ff => true
  | rel _ args => args.all bound.contains
  | eq i j => bound.contains i && bound.contains j
  | neg φ => closedUnder bound φ
  | conj φ ψ | disj φ ψ => closedUnder bound φ && closedUnder bound ψ
  | ex i φ | all i φ => closedUnder (i :: bound) φ

/-- A sentence is a formula without free variables. -/
def isSentence (φ : Formula) : Bool :=
  closedUnder [] φ

/-- Number of nodes of the syntax tree. -/
def size : Formula → Nat
  | tt | ff | rel .. | eq .. => 1
  | neg φ | ex _ φ | all _ φ => size φ + 1
  | conj φ ψ | disj φ ψ => size φ + size ψ + 1

end Formula

end ElementaryTransformer
