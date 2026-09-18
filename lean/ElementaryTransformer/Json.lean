import Lean.Data.Json
import ElementaryTransformer.Certificate

/-!
# Reading certificates from JSON

A certificate is a JSON object

    {"k": 3, "q": 3, "left": S, "right": S, "formula": F}

where a structure `S` is `{"size": n, "relations": [{"name": "E", "arity": 2,
"tuples": [[0, 1], …]}, …]}` and a formula `F` uses the operators `true`,
`false`, `rel` (`symbol`, `args`), `eq` (`left`, `right`), `not` (`body`),
`and`/`or` (`left`, `right`) and `exists`/`forall` (`var`, `body`).
Relations are stored as dense boolean tables, so each atom is decided in time
linear in its arity.
-/

namespace ElementaryTransformer

open Lean (Json)

structure DenseRelation where
  name : String
  arity : Nat
  table : Array Bool

def tableIndex (n : Nat) (args : List Nat) : Nat :=
  args.foldl (fun acc a => acc * n + a) 0

def DenseRelation.holds (r : DenseRelation) (n : Nat) (args : List Nat) : Bool :=
  args.length == r.arity && args.all (· < n) && r.table.getD (tableIndex n args) false

def FinStructure.ofDense (n : Nat) (rels : List DenseRelation) : FinStructure where
  size := n
  rel R args := match rels.find? (·.name == R) with
    | some r => r.holds n args
    | none => false

private def getNat (j : Json) (field : String) : Except String Nat := do
  (← j.getObjVal? field).getNat?

private def parseTuple (j : Json) : Except String (List Nat) := do
  let arr ← j.getArr?
  arr.toList.mapM Json.getNat?

def parseRelation (n : Nat) (j : Json) : Except String DenseRelation := do
  let name ← (← j.getObjVal? "name").getStr?
  let arity ← getNat j "arity"
  let tuples ← (← (← j.getObjVal? "tuples").getArr?).toList.mapM parseTuple
  let mut table := Array.replicate (n ^ arity) false
  for t in tuples do
    unless t.length == arity && t.all (· < n) do
      throw s!"relation {name}: invalid tuple {t}"
    table := table.set! (tableIndex n t) true
  return { name, arity, table }

def parseStructure (j : Json) : Except String FinStructure := do
  let n ← getNat j "size"
  let rels ← (← (← j.getObjVal? "relations").getArr?).toList.mapM (parseRelation n)
  return FinStructure.ofDense n rels

partial def parseFormula (j : Json) : Except String Formula := do
  let op ← (← j.getObjVal? "op").getStr?
  match op with
  | "true" => return .tt
  | "false" => return .ff
  | "rel" =>
    let sym ← (← j.getObjVal? "symbol").getStr?
    return .rel sym (← parseTuple (← j.getObjVal? "args"))
  | "eq" => return .eq (← getNat j "left") (← getNat j "right")
  | "not" => return .neg (← parseFormula (← j.getObjVal? "body"))
  | "and" => return .conj (← parseFormula (← j.getObjVal? "left")) (← parseFormula (← j.getObjVal? "right"))
  | "or" => return .disj (← parseFormula (← j.getObjVal? "left")) (← parseFormula (← j.getObjVal? "right"))
  | "exists" => return .ex (← getNat j "var") (← parseFormula (← j.getObjVal? "body"))
  | "forall" => return .all (← getNat j "var") (← parseFormula (← j.getObjVal? "body"))
  | other => throw s!"unknown operator {other}"

def parseCertificate (j : Json) : Except String Certificate := do
  return {
    left := ← parseStructure (← j.getObjVal? "left")
    right := ← parseStructure (← j.getObjVal? "right")
    formula := ← parseFormula (← j.getObjVal? "formula")
    k := ← getNat j "k"
    q := ← getNat j "q"
  }

end ElementaryTransformer
