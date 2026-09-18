import ElementaryTransformer

/-!
`et-check FILE` reads one JSON certificate per line and prints one JSON result
per line. The exit status is 0 iff every certificate is accepted.
-/

open ElementaryTransformer Lean

def checkLine (index : Nat) (line : String) : Json × Bool :=
  match Json.parse line >>= parseCertificate with
  | .error e => (Json.mkObj [("index", index), ("valid", false), ("error", e)], false)
  | .ok c =>
    let valid := c.check
    (Json.mkObj [
      ("index", index), ("valid", valid),
      ("left", c.formula.eval c.left ρ₀), ("right", c.formula.eval c.right ρ₀),
      ("rank", c.formula.rank), ("sentence", c.formula.isSentence),
      ("within_k", c.formula.usesBelow c.k)], valid)

def main (args : List String) : IO UInt32 := do
  let some path := args.head?
    | IO.eprintln "usage: et-check FILE (one JSON certificate per line)"; return 2
  let lines := (← IO.FS.readFile path).splitOn "\n" |>.filter (fun l => !l.all Char.isWhitespace)
  let mut ok := true
  for (line, i) in lines.zip (List.range lines.length) do
    let (result, valid) := checkLine i line
    IO.println result.compress
    ok := ok && valid
  return if ok then 0 else 1
