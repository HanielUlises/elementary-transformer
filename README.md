# elementary-transformer

Haniel Ulises Vasquez Morales

## Abstract

Whether two finite structures satisfy the same first-order sentences with at most $k$ variables and quantifier rank at most $q$ is decidable by an exhaustive game. The question addressed here is how much of this decision a neural sequence model can acquire from labelled examples, and whether what it acquires carries over to larger structures, to deeper distinctions and to families designed to defeat combinatorial heuristics. This repository provides the data needed to study that question with exact labels. It generates pairs $(\mathfrak A, \mathfrak B)$ of finite relational structures and labels each pair with the least rank $q^\ast$ at which the two structures are separated in $\mathrm{FO}^k$, computed by an exact solver for the $k$-pebble Ehrenfeucht–Fraïssé game; with a sentence of $\mathrm{FO}^k$ of rank $q^\ast$ that is true in $\mathfrak A$ and false in $\mathfrak B$, extracted from a winning strategy of Spoiler and verified by a model checker and, optionally, by a checker written and proved sound in Lean 4; and with the corresponding label in the counting logic $\mathrm{C}^k$ obtained from the Weisfeiler–Leman algorithm. The structures are serialised as sequences of atomic types of $k$-tuples, and the data are split so as to measure generalisation in size, in family and in quantifier rank separately. The scope is deliberately narrow. The pipeline covers finite relational signatures, with graphs and linear orders as the main instances and multi-agent Kripke models as a modal counterpart, and targets the small parameters ($k \le 4$, $q \le 4$, universes of a few dozen elements) at which exact labelling remains cheap.

## Preliminaries

A relational signature $\sigma$ is a finite set of relation symbols, each symbol $R$ carrying an arity $\mathrm{ar}(R) \ge 1$. A finite $\sigma$-structure $\mathfrak A = (A, (R^{\mathfrak A})_{R \in \sigma})$ consists of a finite universe $A$ and relations $R^{\mathfrak A} \subseteq A^{\mathrm{ar}(R)}$. Throughout, $A = [n] = \lbrace 0, \dots, n-1 \rbrace$ and $n = |A|$ is the size of $\mathfrak A$. Simple graphs are structures over one binary symbol $E$ interpreted as a symmetric irreflexive relation, and linear orders are structures over one binary symbol $<$; the linear order with $n$ elements is written $L_n$.

For $k \ge 1$ the logic $\mathrm{FO}^k$ consists of the first-order formulas over $\sigma$ whose variables, free or bound, belong to $\lbrace x_1, \dots, x_k \rbrace$. Atomic formulas are $x_i = x_j$ and $R(x_{i_1}, \dots, x_{i_r})$ with $r = \mathrm{ar}(R)$, and formulas are closed under $\neg$, $\wedge$, $\vee$, $\exists x_i$ and $\forall x_i$. Variables may be requantified, which is what makes $\mathrm{FO}^k$ expressive despite the bound on the number of variables. The quantifier rank is defined by

$$\mathrm{qr}(\alpha) = 0 \text{ for atomic } \alpha, \qquad \mathrm{qr}(\neg\varphi) = \mathrm{qr}(\varphi), \qquad \mathrm{qr}(\varphi \wedge \psi) = \mathrm{qr}(\varphi \vee \psi) = \max(\mathrm{qr}(\varphi), \mathrm{qr}(\psi)),$$

$$\mathrm{qr}(\exists x_i\, \varphi) = \mathrm{qr}(\forall x_i\, \varphi) = \mathrm{qr}(\varphi) + 1,$$

and $\mathrm{FO}^k_q$ denotes the formulas of $\mathrm{FO}^k$ of rank at most $q$. Two structures are $\mathrm{FO}^k_q$-equivalent, written $\mathfrak A \equiv^k_q \mathfrak B$, when they satisfy the same sentences of $\mathrm{FO}^k_q$. The quantity studied in this work is

$$q^\ast_k(\mathfrak A, \mathfrak B) = \min \lbrace q : \mathfrak A \not\equiv^k_q \mathfrak B \rbrace,$$

with $q^\ast_k(\mathfrak A, \mathfrak B) = \bot$ when the set is empty up to the bound $q_{\max}$ fixed for a dataset. Without the superscript, $\equiv_q$ refers to full first-order logic.

The $k$-pebble Ehrenfeucht–Fraïssé game $G^k_q(\mathfrak A, \mathfrak B)$ is played by Spoiler and Duplicator with $k$ pairs of pebbles. A position assigns to some set $P \subseteq [k]$ of pebble indices elements $a_i \in A$ and $b_i \in B$. In each of $q$ rounds Spoiler chooses an index $i \in [k]$ and places pebble $i$ on an element of one of the two structures, removing it from its previous position if it was on the board; Duplicator answers by placing the other copy of pebble $i$ on an element of the other structure. Duplicator wins the play if at the start and after every round the map $a_i \mapsto b_i$ with $i \in P$ is a partial isomorphism, that is, $a_i = a_j \iff b_i = b_j$ and

$$(a_{i_1}, \dots, a_{i_r}) \in R^{\mathfrak A} \iff (b_{i_1}, \dots, b_{i_r}) \in R^{\mathfrak B}$$

for every $R \in \sigma$ and all indices $i_1, \dots, i_r \in P$. Otherwise Spoiler wins.

The counting logic $\mathrm{C}^k$ extends $\mathrm{FO}^k$ by the quantifiers $\exists^{\ge m} x_i\, \varphi$ for every $m \ge 1$, expressing that at least $m$ elements satisfy $\varphi$. Since $\exists$ is $\exists^{\ge 1}$, every sentence of $\mathrm{FO}^k$ is a sentence of $\mathrm{C}^k$, and $\mathrm{C}^k$-equivalence refines $\mathrm{FO}^k$-equivalence at every rank.

For the Weisfeiler–Leman algorithm, write $\mathrm{atp}(\bar a)$ for the atomic type of a tuple, the set of atomic formulas it satisfies, and $\bar a[i/b]$ for the tuple obtained from $\bar a$ by replacing its $i$-th entry by $b$. The $k$-dimensional algorithm colours $A^k$ by

$$c_0(\bar a) = \mathrm{atp}(\bar a), \qquad c_{r+1}(\bar a) = \Big(c_r(\bar a),\ \big\lbrace\!\big\lbrace \big(\mathrm{atp}(\bar a b),\, c_r(\bar a[1/b]), \dots, c_r(\bar a[k/b])\big) : b \in A \big\rbrace\!\big\rbrace\Big),$$

where $\lbrace\!\lbrace \cdot \rbrace\!\rbrace$ is a multiset, and it distinguishes $\mathfrak A$ from $\mathfrak B$ when the multisets of colours of the two structures differ at some round, the colours being named jointly. This is the version of Cai, Fürer and Immerman (1992) of the algorithm introduced by Weisfeiler and Leman (1968); Grohe (2017) gives a systematic account. For graphs and $k \ge 2$ the component $\mathrm{atp}(\bar a b)$ is determined by the others; for $k = 1$ it turns the procedure into colour refinement, and for signatures of higher arity it keeps one refinement step aligned with one round of the bijective pebble game.

A multi-agent Kripke model $M = (W, (R_a)_{a \in \mathrm{Ag}}, V)$ has a finite set of worlds $W$, an accessibility relation $R_a \subseteq W \times W$ for every agent and a valuation $V : W \to 2^{\mathrm{Prop}}$. The relation of $n$-bisimilarity between pointed models is defined by induction. $(M, w) \leftrightarrow_0 (M', w')$ holds when $V(w) = V'(w')$, and $(M, w) \leftrightarrow_{n+1} (M', w')$ holds when $(M, w) \leftrightarrow_0 (M', w')$ and for every agent $a$ each $R_a$-successor of $w$ is $n$-bisimilar to some $R'_a$-successor of $w'$ and conversely.

## Theoretical background

The design of the pipeline rests on the following results. Propositions 1 to 7 are classical and are quoted with references; Propositions 8 and 9 are elementary consequences that justify the solver and the extraction of formulas, and their proofs are sketched.

**Proposition 1** (Ehrenfeucht 1961; Fraïssé 1954; Immerman 1982; see also Ebbinghaus and Flum 1999). For $\sigma$-structures $\mathfrak A$, $\mathfrak B$ and integers $k \ge 1$, $q \ge 0$, $\mathfrak A \equiv^k_q \mathfrak B$ holds if and only if Duplicator has a winning strategy in $G^k_q(\mathfrak A, \mathfrak B)$. Moreover every first-order sentence of rank $q$ is equivalent to a sentence of $\mathrm{FO}^q_q$, obtained by renaming each bound variable after the depth of its quantifier, so that $\equiv^q_q$ coincides with $\equiv_q$.

The second statement is what allows the solver, which works with a fixed number of pebbles, to reproduce results about full first-order logic. With $k \ge q$ pebbles, $q^\ast_k$ agrees with the first-order rank of separation whenever the latter is at most $q$.

**Proposition 2** (Rosenstein 1982; Libkin 2004). For $m, n \ge 1$ and $q \ge 1$, $L_m \equiv_q L_n$ holds if and only if $m = n$ or $m, n \ge 2^q - 1$. Consequently, for $m \ne n$,

$$q^\ast(L_m, L_n) = \lfloor \log_2(\min(m, n) + 1) \rfloor + 1.$$

Linear orders thus provide pairs whose separating rank is known in closed form and grows logarithmically with the size, which makes them the reference family for testing the solver and for populating every rank stratum.

**Proposition 3** (Glebskii, Kogan, Liogon'kii and Talanov 1969; Fagin 1976). For every first-order sentence $\varphi$ over graphs, the probability that the uniform random graph $G(n, 1/2)$ satisfies $\varphi$ converges to 0 or to 1 as $n \to \infty$.

**Proposition 4**. For fixed $k$ and $q$, if $\mathfrak A_n$ and $\mathfrak B_n$ are independent copies of $G(n, 1/2)$, then $\Pr[\mathfrak A_n \equiv^k_q \mathfrak B_n] \to 1$.

*Proof.* Over a finite relational signature there are finitely many sentences of rank at most $q$ up to logical equivalence, so $\equiv_q$ has finitely many classes, each defined by a single sentence. By Proposition 3 each class has limit probability 0 or 1, and since the probabilities sum to 1 exactly one class has limit 1. Both graphs fall in that class with probability tending to 1, and $\equiv_q$ refines $\equiv^k_q$. $\square$

Proposition 4 explains why independent dense random graphs are useless as pairs, since their labels become trivial as the size grows. The pipeline therefore pairs a random graph with a minimal mutation of itself, and it also samples the sparse regime $p = c/n$, where the probability of every first-order sentence still converges (Lynch 1992) but the limits are in general strictly between 0 and 1, so that non-trivial labels persist.

**Proposition 5** (Immerman and Lander 1990; Cai, Fürer and Immerman 1992). For $k \ge 1$ and graphs $\mathfrak A$, $\mathfrak B$, the $k$-dimensional Weisfeiler–Leman algorithm distinguishes $\mathfrak A$ from $\mathfrak B$ if and only if $\mathfrak A \not\equiv_{\mathrm{C}^{k+1}} \mathfrak B$.

The same equivalence holds for arbitrary signatures of arity at most $k+1$ with the refinement given in the preliminaries, since one step of that refinement corresponds to one round of the bijective $(k+1)$-pebble game of Hella (1996). Proposition 5 is used in two ways. It provides the $\mathrm{C}^k$ labels, computed by the $(k-1)$-dimensional algorithm, and it provides a consistency check, because a pair that is $\mathrm{C}^k$-equivalent must be $\mathrm{FO}^k$-equivalent at every rank.

**Proposition 6** (Cai, Fürer and Immerman 1992). For every $k$ there are non-isomorphic 3-regular graphs with $O(k)$ vertices that are $\mathrm{C}^k$-equivalent. Such pairs are obtained by replacing every vertex of a suitable 3-regular base graph $G$ by a gadget and connecting the gadgets along the edges of $G$, either directly or with one twisted connection; the two resulting graphs $X(G)$ and $\tilde X(G)$ are not isomorphic when $G$ is connected.

**Proposition 7** (Blackburn, de Rijke and Venema 2001). For finitely many agents and propositional letters, $(M, w) \leftrightarrow_n (M', w')$ holds if and only if $w$ and $w'$ satisfy the same multi-modal formulas of modal depth at most $n$. Every modal formula of depth $n$ is equivalent, through the standard translation, to a formula of $\mathrm{FO}^2$ of rank $n$ with one free variable.

The solver does not search the space of positions of the game directly. It relies on the following characterisation, in which unplaced pebbles are represented by a symbol $\bot$, and $\mathrm{atp}(\bar a)$ for $\bar a \in (A \cup \lbrace \bot \rbrace)^k$ records which entries are placed together with the atomic type of the placed ones.

**Proposition 8** (type refinement). For $\bar a \in (A \cup \lbrace \bot \rbrace)^k$ define

$$\mathrm{tp}_0(\bar a) = \mathrm{atp}(\bar a), \qquad \mathrm{tp}_{r+1}(\bar a) = \Big(\mathrm{tp}_r(\bar a),\ \big(\lbrace \mathrm{tp}_r(\bar a[i/a]) : a \in A \rbrace\big)_{i \in [k]}\Big),$$

and define $\mathrm{tp}_r$ on $(B \cup \lbrace \bot \rbrace)^k$ in the same way. Then Duplicator wins the $r$-round game from the position $(\bar a, \bar b)$ if and only if $\mathrm{tp}_r(\bar a) = \mathrm{tp}_r(\bar b)$. In particular $q^\ast_k(\mathfrak A, \mathfrak B)$ is the least $r$ at which the types of the empty configurations $\bot^k$ differ.

*Proof sketch.* For $r = 0$ both sides say that the pebbled elements form a partial isomorphism. For $r + 1$, Duplicator wins if and only if the position is a partial isomorphism and, for every pebble $i$ and every element $a$ chosen by Spoiler in $\mathfrak A$, there is an answer $b$ with Duplicator winning $r$ rounds from $(\bar a[i/a], \bar b[i/b])$, and symmetrically for moves in $\mathfrak B$. By induction this says that the sets of round-$r$ types reachable by moving pebble $i$ coincide on both sides, which is the definition of equal round-$(r+1)$ types. $\square$

**Proposition 9** (extraction of distinguishing formulas). Suppose $\mathrm{tp}_r(\bar a) \ne \mathrm{tp}_r(\bar b)$ and $r$ is least with this property. If $r = 0$ some literal $\lambda$ over the placed pebbles holds at $\bar a$ and fails at $\bar b$. If $r > 0$ there are a pebble $i$ and, say, an element $a \in A$ whose type $\mathrm{tp}_{r-1}(\bar a[i/a])$ differs from $\mathrm{tp}_{r-1}(\bar b[i/b])$ for every $b \in B$. Grouping the answers $b$ into classes $\tau$ of equal type $\mathrm{tp}_{r-1}(\bar b[i/b])$ and choosing recursively formulas $\psi_\tau$ that hold at $\bar a[i/a]$ and fail on class $\tau$, the formula

$$\varphi = \exists x_i \bigwedge_{\tau} \psi_\tau$$

belongs to $\mathrm{FO}^k$, has rank $r$, holds at $\bar a$ and fails at $\bar b$. When Spoiler's move is in $\mathfrak B$ the dual construction $\forall x_i \bigvee_\sigma \chi_\sigma$ is used.

*Proof sketch.* The witness $a$ satisfies every conjunct, so $\varphi$ holds at $\bar a$. Every $b \in B$ lies in some class $\tau$, and since formulas of rank at most $r - 1$ cannot tell apart configurations with equal $\mathrm{tp}_{r-1}$ (Propositions 1 and 8), the conjunct $\psi_\tau$ fails at $\bar b[i/b]$. The rank bound follows by induction, and the minimality of $r$ forces equality. $\square$

## Generation and labelling procedure

**Families of structures.** Linear orders $L_n$ are produced directly. Cycles $C_n$ and disjoint unions of cycles yield pairs such as $C_6$ against $C_3 \uplus C_3$, which colour refinement cannot separate but three variables can. Random graphs are drawn from $G(n, 1/2)$ and from the sparse model $G(n, c/n)$ with $c \in \lbrace 1, 2, 3 \rbrace$. Random $d$-regular graphs with $d \in \lbrace 2, 3, 4 \rbrace$ are drawn from the pairing model conditioned on simplicity, which is uniform on labelled simple $d$-regular graphs (Bollobás 1980); two regular graphs of the same degree are never separated by colour refinement. When nauty's `geng` is installed (McKay and Piperno 2014), the complete list of graphs on at most nine vertices up to isomorphism is available as a further family. The graphs of Cai, Fürer and Immerman are built from the construction of Proposition 6 over the connected 3-regular base graphs $K_4$, $K_{3,3}$, the triangular prism, the cube and the Petersen graph, which gives 3-regular graphs with ten vertices per base vertex. The implementation is uncoloured, so non-isomorphism is not inherited from the coloured construction; it is certified instead by the 3-dimensional Weisfeiler–Leman algorithm, which separates $X(G)$ from $\tilde X(G)$ for the bases used while the 2-dimensional algorithm does not. The test suite also checks through an explicit isomorphism that two twisted edges give a graph isomorphic to $X(G)$. Strongly regular graphs enter as pairs with equal parameters, which the 2-dimensional algorithm never separates. The $4 \times 4$ rook's graph and the Shrikhande graph share the parameters $(16, 6, 2, 2)$, and the triangular graph $T(8)$ and the three Chang graphs, obtained from $T(8)$ by Seidel switching, share the parameters $(28, 12, 6, 4)$ (Brouwer and Van Maldeghem 2022). These graphs are constructed without external software and their parameters are verified by the tests. When SageMath is installed, further strongly regular graphs can be retrieved from its constructions.

**Construction of pairs.** Three constructions are used. A minimal mutation pairs a structure with the result of toggling one pair of vertices or of a double edge swap that replaces edges $\lbrace a, b \rbrace, \lbrace c, d \rbrace$ by $\lbrace a, d \rbrace, \lbrace c, b \rbrace$ and so preserves the degree sequence. An isomorphic pair applies a uniformly random permutation, so its label is $\bot$ by construction and serves as a control. A pair from a hard family is one of the constructions above, such as $X(G)$ against $\tilde X(G)$ or two strongly regular graphs with equal parameters. Before labelling, both structures are relabelled by independent random permutations and the order of the pair is randomised. Without this step a mutated pair would share its labelling with the original up to one entry, and a sequence model could decide non-equivalence by aligning tokens instead of reasoning about the structures. All randomness is derived from a single seed through the splittable generator of JAX, so every example can be regenerated from the seed and its index.

**Computation of $q^\ast$.** The solver implements Proposition 8 in C++. For each structure it enumerates the configurations $(A \cup \lbrace \bot \rbrace)^k$, encoded as integers in base $n + 1$, and computes the types round by round, naming them jointly for both structures through exact interning of integer rows, so that equal names mean equal types and no hash collision can merge distinct types. Since the set attached to a configuration $\bar a$ and a pebble $i$ does not depend on the current position of pebble $i$, it is computed once per configuration with pebble $i$ lifted. The computation stops at the first round in which the empty configurations of the two structures receive different types, which gives $q^\ast_k$, or when the joint partition stops refining, in which case the structures are $\mathrm{FO}^k$-equivalent at every rank, or at $q_{\max}$. With $n = \max(|A|, |B|)$ and $s = \sum_{R \in \sigma} k^{\mathrm{ar}(R)}\, \mathrm{ar}(R)$, the atomic types cost $O((n+1)^k (k^2 + s))$ and each further round costs $O(k (n+1)^k \log n)$ expected time, so that

$$T(n, k, q) = O\big((n+1)^k (k^2 + s) + q\, k\, (n+1)^k \log n\big), \qquad M(n, k, q) = O\big(q\, (n+1)^k\big).$$

A direct search over pairs of configurations has $(n^2 + 1)^k$ positions with $O(k n^2)$ successors each, so it takes $O(q k n^{2k+2})$ time in the worst case. The type tables cut the exponent roughly in half and make $k = 3$, $q = 4$ and $n = 50$ a matter of milliseconds. The test suite compares the solver with such a direct memoised search on small random structures.

**Extraction and verification of formulas.** The tables are also a complete description of a winning strategy for Spoiler. At a position whose types first differ at round $r$, Spoiler plays a move as in Proposition 9, choosing among the admissible moves one that minimises the number of classes of answers. The strategy is available as an object that returns Spoiler's move at any position reached against any Duplicator, as an explicit tree with the answers grouped into classes, and as the distinguishing sentence of Proposition 9. Conjunctions and disjunctions are pruned greedily: a class of answers is skipped when a conjunct already built fails on its representative, which is sound because all members of a class have the same type. Every sentence is then verified before it is stored. A vectorised model checker evaluates it bottom-up in time $O(|\varphi|\, n^k)$ and confirms that it holds in $\mathfrak A$ and fails in $\mathfrak B$, that its rank equals $q^\ast$ and that it uses at most $k$ variables. Optionally, the sentence is also checked by a program written in Lean 4 (de Moura and Ullrich 2021), whose evaluator is proved to agree with the Tarskian satisfaction relation and whose acceptance is proved to imply the semantic claim (`Formula.eval_iff_sat`, `Formula.sentence_sat_iff` and `Certificate.check_sound` in `lean/ElementaryTransformer`). The $\mathrm{C}^k$ label is computed with the $(k-1)$-dimensional algorithm, implemented in JAX with exact joint renaming of colours, and the pipeline aborts if a pair labelled $\mathrm{C}^k$-equivalent were separated in $\mathrm{FO}^k$. The test suite additionally plays every extracted strategy against all Duplicator answers on small instances and checks, without using the tables, that Spoiler always wins within $q^\ast$ rounds.

**Modal counterpart.** Random multi-agent Kripke models are generated with independent accessibility edges and valuation bits, and $n$-bisimilarity is computed by partition refinement on the disjoint union of the two models. The successor sets of every world are obtained as a product of the accessibility matrices with the one-hot colouring. Through Proposition 7, a pair of pointed models separated at modal depth $d$ is separated in $\mathrm{FO}^2$ at rank at most $d + 1$ once the distinguished world is marked by a unary predicate, and the tests verify this bound against the game solver.

## Data format

**Definition 1** (atomic type vocabulary). Fix a signature $\sigma$ and $k \ge 1$. A consistent atomic $k$-type is determined by a set partition $\pi$ of $[k]$, the equality pattern, together with a truth value for every block-level atom, meaning every $R(y_1, \dots, y_r)$ whose arguments are representatives of blocks of $\pi$. The representative of a block is its least element, and atoms whose value is forced by a declared symmetry or irreflexivity are omitted. Let $\beta(\pi)$ be the number of remaining atoms, and order the partitions lexicographically by their restricted growth strings. The code of a type with partition $\pi$ and atom bits $e_0, \dots, e_{\beta(\pi)-1}$, listed in lexicographic order of their argument tuples, is

$$\iota = \sum_{\pi' < \pi} 2^{\beta(\pi')} + \sum_{j < \beta(\pi)} e_j\, 2^j,$$

and the vocabulary $V_{\sigma,k}$ has $|V_{\sigma,k}| = \sum_{\pi} 2^{\beta(\pi)}$ codes, which are in bijection with the consistent atomic $k$-types. For simple graphs $\beta(\pi) = \binom{|\pi|}{2}$, so that $|V_{\sigma,3}| = 15$ and $|V_{\sigma,4}| = 127$.

**Definition 2** (tokenisation of a structure). The token sequence of a $\sigma$-structure $\mathfrak A$ with universe $[n]$ is the sequence $\tau_k(\mathfrak A) \in V_{\sigma,k}^{\,n^k}$ given by

$$\tau_k(\mathfrak A)_{\mathrm{idx}(\bar a)} = \iota\big(\mathrm{atp}(\bar a)\big), \qquad \mathrm{idx}(a_1, \dots, a_k) = \sum_{j=1}^{k} a_j\, n^{k-j},$$

so that the positions enumerate $A^k$ in lexicographic order and the sequence can equivalently be read as a tensor of shape $n \times \dots \times n$. The tokenisation is equivariant: if $\mathfrak A'$ is the image of $\mathfrak A$ under a permutation $\rho$ of $[n]$, then $\tau_k(\mathfrak A')_{\mathrm{idx}(\rho \bar a)} = \tau_k(\mathfrak A)_{\mathrm{idx}(\bar a)}$. When a dataset contains several signatures, such as graphs and linear orders, their vocabularies are concatenated and each block is shifted by the size of the preceding ones.

**Definition 3** (formula tokens). A formula is serialised in prefix notation with binary connectives over the tokens `TRUE`, `FALSE`, `NOT`, `AND`, `OR`, `EXISTS`, `FORALL`, `EQ`, `R:`$R$ for $R \in \sigma$ and `x`$i$ for $i < k$. The arity of each relation symbol is read from the signature, so the notation is unambiguous and every well-formed sequence decodes to a unique formula.

**Definition 4** (example). An example is a record consisting of the pair of structures, both token sequences, the value $q^\ast_k \in \lbrace 1, \dots, q_{\max} \rbrace \cup \lbrace \bot \rbrace$, a flag indicating whether the joint type partition stabilised, the distinguishing sentence in tree and token form when $q^\ast_k \ne \bot$, Spoiler's first move, the $\mathrm{C}^k$ label with the Weisfeiler–Leman round at which it was decided, the verdict of the Lean checker when it was run, and the family, construction and parameters of the pair. A dataset directory contains a `manifest.json` with the configuration, the vocabularies and per-split statistics, and one directory per split with shards. Each shard stores the arrays in a compressed `.npz` file, with concatenated tokens and offsets, and the remaining fields in a `.jsonl` file with one line per example.

## Experimental protocol

The data are divided into five splits, each designed to isolate one kind of generalisation. The `train` and `val` splits contain structures with $n \le 12$ from the in-distribution families (linear orders, cycles, dense and sparse random graphs, regular graphs and, when available, the exhaustive family) and only pairs with $q^\ast \le q_{\max} - 1$ or $q^\ast = \bot$. Every test split changes exactly one of these three conditions.

The size split `test_size` keeps the families and the admissible ranks but draws structures with $20 \le n \le 50$. A sentence of fixed rank expresses a property that does not depend on the size of the structure, so a model that has learned the logic, rather than statistics of small structures, should transfer. Proposition 4 warns that the label distribution itself moves with $n$, since larger random structures are equivalent at small ranks more often, which is why the results must be read per stratum of $q^\ast$ and not only in aggregate.

The family split `test_family` contains only CFI graphs and strongly regular graphs, which never occur in training, together with mutations and isomorphic copies of them. These are the classical hard instances for combinatorial invariants. Their paired members are regular, cospectral in the strongly regular case, and indistinguishable by the Weisfeiler–Leman algorithm of low dimension, so a model that relies on degree sequences, spectra or colour refinement will fail on them. The non-isomorphic pairs of this split are $\mathrm{C}^3$-equivalent, hence $\mathrm{FO}^3$-equivalent at every rank, and with $k \le 3$ the split therefore also measures whether a model confuses logical equivalence with isomorphism. The split necessarily changes the size as well, since the smallest pairs have 16 vertices and CFI graphs have ten vertices per base vertex, and this confound should be kept in mind.

The rank split `test_rank` keeps the families and the small sizes of `train` but contains only pairs with $q^\ast = q_{\max}$. With $q_{\max} = 4$ the model is trained on separations of depth at most 3 and evaluated on separations of depth 4, which it has never observed. Since the type refinement of Proposition 8 needs one round per unit of rank, this split probes whether a model with a fixed number of layers has learned an iterative procedure or a bounded-depth one.

Within every split the examples are stratified by $q^\ast$, with equal quotas for the ranks allowed in the split and for $\bot$. The quotas start at rank 2 whenever every relation symbol is irreflexive and binary, as for graphs and linear orders. In that case a sentence of rank 1 is a Boolean combination of sentences $\exists x\, \alpha(x)$ with $\alpha$ quantifier-free in one variable, and it can only express that the universe is non-empty. When half of the sampling budget has been spent, strata that have received no example are dropped and their quota is redistributed, and the manifest records both the dropped strata and the final counts. Samples in `val` are drawn independently of `train` but from the same distribution, so isomorphic duplicates across these two splits are possible for the small families. The three test splits are disjoint from `train` by construction, through size, family or label.

## Usage

The package requires Python 3.10 or later, a C++17 compiler and CMake. The Lean checker requires `elan`, and `geng` and SageMath are optional.

```sh
python -m venv .venv && . .venv/bin/activate
pip install scikit-build-core pybind11
pip install --no-build-isolation -e ".[test]"
lake build                      # Lean library and the et-check executable
pytest                          # Python tests, including Lean certificates when et-check is built
cmake -S . -B build/cpp && cmake --build build/cpp && ctest --test-dir build/cpp
```

The command-line interface has four subcommands. `generate` builds a dataset with the protocol above, `solve` runs the game solver on two structures given by name, by graph6 string or by file, `wl` runs the Weisfeiler–Leman algorithm, and `inspect` summarises a dataset.

```sh
elementary-transformer solve cycle:6 cycles:3,3 --k 3 --q 4 --lean
elementary-transformer solve order:7 order:8 --k 4 --q 4
elementary-transformer wl rook:4 shrikhande --dim 2
elementary-transformer generate --out data/k3q4 --k 3 --q 4 --train 20000 --val 2000 --test 1000 --lean-verify
elementary-transformer inspect data/k3q4
```

The prototype that preceded this project, a pipeline for link prediction with a Lean verifier, is preserved in `legacy/` as an independent Lake project.

## References

Blackburn, P., de Rijke, M. and Venema, Y. (2001). *Modal Logic*. Cambridge Tracts in Theoretical Computer Science 53. Cambridge University Press.

Bollobás, B. (1980). A probabilistic proof of an asymptotic formula for the number of labelled regular graphs. *European Journal of Combinatorics*, 1(4), 311–316.

Brouwer, A. E. and Van Maldeghem, H. (2022). *Strongly Regular Graphs*. Encyclopedia of Mathematics and its Applications 182. Cambridge University Press.

Cai, J.-Y., Fürer, M. and Immerman, N. (1992). An optimal lower bound on the number of variables for graph identification. *Combinatorica*, 12(4), 389–410.

Ebbinghaus, H.-D. and Flum, J. (1999). *Finite Model Theory*, 2nd edition. Perspectives in Mathematical Logic. Springer.

Ehrenfeucht, A. (1961). An application of games to the completeness problem for formalized theories. *Fundamenta Mathematicae*, 49, 129–141.

Fagin, R. (1976). Probabilities on finite models. *Journal of Symbolic Logic*, 41(1), 50–58.

Fraïssé, R. (1954). Sur quelques classifications des systèmes de relations. *Publications Scientifiques de l'Université d'Alger, Série A*, 1, 35–182.

Glebskii, Y. V., Kogan, D. I., Liogon'kii, M. I. and Talanov, V. A. (1969). Range and degree of realizability of formulas in the restricted predicate calculus. *Cybernetics*, 5(2), 142–154.

Grohe, M. (2017). *Descriptive Complexity, Canonisation, and Definable Graph Structure Theory*. Lecture Notes in Logic 47. Cambridge University Press.

Hella, L. (1996). Logical hierarchies in PTIME. *Information and Computation*, 129(1), 1–19.

Immerman, N. (1982). Upper and lower bounds for first order expressibility. *Journal of Computer and System Sciences*, 25(1), 76–98.

Immerman, N. and Lander, E. (1990). Describing graphs: a first-order approach to graph canonization. In A. L. Selman (ed.), *Complexity Theory Retrospective*, 59–81. Springer.

Libkin, L. (2004). *Elements of Finite Model Theory*. Texts in Theoretical Computer Science. Springer.

Lynch, J. F. (1992). Probabilities of sentences about very sparse random graphs. *Random Structures & Algorithms*, 3(1), 33–53.

McKay, B. D. and Piperno, A. (2014). Practical graph isomorphism, II. *Journal of Symbolic Computation*, 60, 94–112.

de Moura, L. and Ullrich, S. (2021). The Lean 4 theorem prover and programming language. In *Automated Deduction, CADE 28*, Lecture Notes in Computer Science 12699, 625–635. Springer.

Rosenstein, J. G. (1982). *Linear Orderings*. Pure and Applied Mathematics 98. Academic Press.

Weisfeiler, B. and Leman, A. (1968). The reduction of a graph to canonical form and the algebra which appears therein. *Nauchno-Technicheskaya Informatsia*, Series 2, 9, 12–16.
