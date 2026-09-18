import ElementaryTransformer.Certificate

/-!
# Examples checked by the kernel

Two small separations that the Python pipeline also produces: the orders
`L₂` and `L₃` are separated at quantifier rank 2 with two variables, and the
cycle `C₆` and the disjoint union `C₃ ⊎ C₃` are separated at rank 3 with three
variables. The certificates are decided by `decide` and then turned into
statements about `Sat` by `Certificate.check_sound`.
-/

namespace ElementaryTransformer.Examples

open Formula

def linearOrder (n : Nat) : FinStructure where
  size := n
  rel R args := R == "<" && match args with
    | [a, b] => decide (a < b)
    | _ => false

/-- A cycle on `m` vertices, or `copies` disjoint cycles on `m` vertices each. -/
def cycles (copies m : Nat) : FinStructure where
  size := copies * m
  rel R args := R == "E" && match args with
    | [a, b] => a / m == b / m && ((a + 1) % m == b % m || (b + 1) % m == a % m)
    | _ => false

/-- "Some element has a predecessor and a successor." -/
def hasMiddle : Formula :=
  ex 0 (conj (ex 1 (rel "<" [1, 0])) (ex 1 (rel "<" [0, 1])))

/-- "There is a triangle." -/
def triangle : Formula :=
  ex 0 (ex 1 (ex 2 (conj (rel "E" [0, 1]) (conj (rel "E" [1, 2]) (rel "E" [0, 2])))))

def orders : Certificate := ⟨linearOrder 3, linearOrder 2, hasMiddle, 2, 2⟩

def triangles : Certificate := ⟨cycles 2 3, cycles 1 6, triangle, 3, 3⟩

theorem orders_check : orders.check = true := by decide

theorem triangles_check : triangles.check = true := by decide

theorem L3_not_equiv_L2 :
    (∀ ρ, hasMiddle.Sat (linearOrder 3) ρ) ∧ (∀ ρ, ¬ hasMiddle.Sat (linearOrder 2) ρ) ∧
      hasMiddle.rank = 2 :=
  let h := orders.check_sound orders_check
  ⟨h.2.2.2.1, h.2.2.2.2, rfl⟩

theorem two_triangles_not_equiv_hexagon :
    (∀ ρ, triangle.Sat (cycles 2 3) ρ) ∧ (∀ ρ, ¬ triangle.Sat (cycles 1 6) ρ) ∧
      triangle.rank = 3 :=
  let h := triangles.check_sound triangles_check
  ⟨h.2.2.2.1, h.2.2.2.2, rfl⟩

end ElementaryTransformer.Examples
