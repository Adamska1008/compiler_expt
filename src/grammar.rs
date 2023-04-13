use crate::symbol::{FirstSet, FollowSet, GrammarSymbol, Production};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Grammar {
    pub non_terminal: HashSet<GrammarSymbol>,
    pub terminal: HashSet<GrammarSymbol>,
    pub productions: Vec<Production>,
    pub first_map: HashMap<GrammarSymbol, FirstSet>,
    pub follow_map: HashMap<GrammarSymbol, FollowSet>,
}

impl Grammar {
    /// Return a new Grammar structure.
    ///
    /// Automatically extract symbols from productions.
    /// Do not build two maps initially. They will be dynamically built afterwards.
    pub fn new(productions: Vec<Production>) -> Self {
        let mut grammar = Grammar {
            non_terminal: HashSet::new(),
            terminal: HashSet::new(),
            productions,
            first_map: HashMap::new(),
            follow_map: HashMap::new(),
        };
        // extract symbols from productions
        for production in &grammar.productions {
            grammar.non_terminal.insert(production.left.clone());
            for symbol in &production.right {
                match symbol {
                    GrammarSymbol::NonTerminalSymbol(_) => {
                        grammar.non_terminal.insert(symbol.clone());
                    }
                    GrammarSymbol::TerminalSymbol(_) => {
                        grammar.terminal.insert(symbol.clone());
                    }
                    _ => {}
                }
            }
        }
        grammar
    }

    /// Return the production whose left is 'symbol'
    ///
    /// # Params
    /// + `symbol`: the left of production
    ///
    /// # Panics
    /// `symbol` is not non terminal
    ///
    /// # Example
    /// ```rust
    /// use compiler_expt::grammar::Grammar;
    /// use compiler_expt::symbol::Production;
    ///
    /// let grammar = Grammar::new(vec![Production::like('A', "ab"), Production::like('B', "bc")]);
    /// assert_eq!(grammar.productions_of(&'A'.into()), vec![Production::like('A', "ab")]);
    /// ```
    pub fn productions_of(&self, symbol: &GrammarSymbol) -> Vec<Production> {
        if !symbol.is_non_terminal() {
            panic!("The symbol {:?} is not non terminal.", symbol);
        }
        self.productions
            .iter()
            .filter(|p| p.left == *symbol)
            .map(|p| p.clone())
            .collect()
    }

    /// Return the first set of the symbol
    ///
    /// # panics
    /// If the symbol is non terminal and is not in the grammar, panics.
    ///
    pub fn first(&mut self, symbol: impl Into<GrammarSymbol>) -> FirstSet {
        let symbol = symbol.into();
        if let GrammarSymbol::NonTerminalSymbol(_) = symbol {
            if !self.non_terminal.contains(&symbol) {
                panic!("The symbol {:?} is not in the grammar", symbol);
            }
            match self.first_map.get(&symbol) {
                Some(first_set) => first_set.clone(),
                None => {
                    let productions: Vec<_> = self
                        .productions
                        .iter()
                        .filter(|&p| p.left == symbol)
                        .map(|p| p.clone())
                        .collect();
                    let mut first_set = FirstSet::new();
                    for p in productions {
                        if p.is_null() {
                            first_set.insert(GrammarSymbol::Null);
                            continue;
                        }
                        first_set.extend(self.first_of_production(p.clone()));
                    }
                    self.first_map.insert(symbol, first_set.clone());
                    first_set
                }
            }
        } else {
            return FirstSet::from([symbol]);
        }
    }

    /// Return the first set of production that in the environment of this grammar.
    pub fn first_of_production(&mut self, production: Production) -> FirstSet {
        let mut res = FirstSet::new();
        if production.is_null() {
            res.insert(GrammarSymbol::Null);
            return res;
        }
        let mut k = 0;
        let n = production.right.len();
        while k < n {
            let first_x_k = self.first(production.right[k].clone());
            res.extend(first_x_k.clone());
            res.remove(&GrammarSymbol::Null);
            if !first_x_k.contains(&GrammarSymbol::Null) {
                break;
            }
            k += 1;
        }
        if k == n {
            res.insert(GrammarSymbol::Null);
        }
        res
    }

    /// Return the follow set of non_terminal symbol.
    ///
    /// # panics
    /// If the symbol is not non_terminal, panics
    pub fn follow(&mut self, symbol: impl Into<GrammarSymbol>) -> FollowSet {
        let symbol = symbol.into();
        if !symbol.is_non_terminal() {
            panic!("The symbol {:?} isn't non terminal symbol", symbol);
        }
        match self.follow_map.get(&symbol) {
            Some(follow_set) => follow_set.clone(),
            None => {
                let mut res = FollowSet::new();
                for production in self.productions.clone() {
                    if let Some(pos) = production.find(&symbol) {
                        if pos == production.right.len() - 1 {
                            // if the symbol is in the end of production
                            if let Some(follow_left) = self.follow_map.get(&production.left) {
                                res.extend(follow_left.clone());
                            } else {
                                let follow_left = self.follow(production.left.clone());
                                self.follow_map.insert(production.left.clone(), follow_left.clone());
                                res.extend(follow_left);
                            }
                        } else {
                            let mut first_beta = self.first(production.right[pos + 1].clone());
                            first_beta.remove(&GrammarSymbol::Null);
                            res.extend(first_beta);
                            // 先跳过beta推导出epsilon的情况
                        }
                    }
                }
                self.follow_map.insert(symbol, res.clone());
                res
            }
        }
    }
}

pub mod ll1 {
    use crate::symbol::{has_left_recursion, GrammarSymbol, Production};

    use super::Grammar;

    #[derive(Debug)]
    // Cell of LL Parsing Table
    pub struct Cell {
        pub stack_top: GrammarSymbol,
        pub input: GrammarSymbol,
        pub production: Production,
    }

    impl Cell {
        fn new(stack_top: GrammarSymbol, input: GrammarSymbol, production: Production) -> Self {
            Self {
                stack_top,
                input,
                production,
            }
        }
    }

    /// Test whether the grammar is ll1
    /// 1. does not have left recursion
    /// 2. for A-> a1|a2|...|an|, FIRST(ai) and FIRST(aj) = empty
    /// 3. if A->e, for a in a1|a2..., FIRST(ai) and FOLLOW(A) = empty
    ///
    /// # Params
    /// + `grammar` grammar
    pub fn is_ll1(grammar: &mut Grammar) -> bool {
        if has_left_recursion(&grammar.productions) {
            return false;
        }
        for non_terminal in grammar.non_terminal.clone() {
            let productions = grammar.productions_of(&non_terminal);
            let len = productions.len();
            for i in 0..len {
                for j in (i + 1)..len {
                    let first_i = grammar.first_of_production(productions[i].clone());
                    let first_j = grammar.first_of_production(productions[j].clone());
                    if !first_i.is_disjoint(&first_j) {
                        return false;
                    }
                }
            }
            if productions.iter().filter(|p| p.is_null()).count() > 0 {
                let follow = grammar.follow(non_terminal.clone());
                for production in &productions {
                    let first = grammar.first_of_production(production.clone());
                    if !follow.is_disjoint(&first) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    ///
    pub fn to_ll1_parse_table(grammar: &mut Grammar) -> Vec<Cell> {
        let mut res = Vec::new();
        for non_terminal in grammar.non_terminal.clone() {
            let productions = grammar.productions_of(&non_terminal);
            for production in productions {
                let first = grammar.first_of_production(production.clone());
                for symbol in first {
                    res.push(Cell::new(non_terminal.clone(), symbol, production.clone()));
                }
                if production.is_null() {
                    let follow = grammar.follow(non_terminal.clone());
                    for symbol in follow {
                        res.push(Cell::new(non_terminal.clone(), symbol, production.clone()));
                    }
                }
            }
        }
        res
    }
}

pub mod lr0 {
    use crate::grammar::Grammar;
    use std::fmt::{Debug, Formatter};
    use std::{
        collections::{HashMap, HashSet},
        hash::Hash,
    };

    use crate::symbol::{GrammarSymbol, Production};

    #[derive(PartialEq, Eq, Hash, Clone)]
    pub struct LR0Item {
        production: Production,
        dot_pos: usize,
    }

    impl Debug for LR0Item {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            let left = self.production.left.inner();
            let mut right: String = String::new();
            let mut cnt = 0;
            for str in self.production.right.iter().map(|s| s.inner()) {
                if cnt == self.dot_pos {
                    right.push_str("·");
                }
                cnt += 1;
                right += &str;
            }
            if cnt == self.dot_pos {
                right.push_str("·");
            }
            write!(f, "LR0Item {{ {}->{} }}", left, right)
        }
    }

    impl LR0Item {
        pub fn new(production: Production, dot_pos: usize) -> Self {
            Self { production, dot_pos }
        }

        pub fn like(left: char, right: &str, dot_pos: usize) -> Self {
            Self {
                production: Production::like(left, right),
                dot_pos,
            }
        }

        /// Check the next symbol after dot
        /// peek(A->a·b) is Some('b')
        pub fn peek(&self) -> Option<&GrammarSymbol> {
            self.production.right_nth(self.dot_pos)
        }

        /// move the dot to next
        pub fn next(&mut self) {
            self.dot_pos += 1;
        }

        pub fn is_accepted(&self) -> bool {
            self.dot_pos == self.production.right.len() && self.production.left.is_start()
        }

        pub fn is_shift(&self) -> bool {
            self.dot_pos < self.production.right.len() && self.production.right[self.dot_pos].is_terminal()
        }

        pub fn is_reduction(&self) -> bool {
            self.dot_pos == self.production.right.len() && !self.production.left.is_start()
        }
    }

    #[derive(Debug, Eq, PartialEq, Clone)]
    pub struct ItemSet {
        pub item_set: HashSet<LR0Item>,
    }

    impl<const N: usize> From<[LR0Item; N]> for ItemSet {
        fn from(items: [LR0Item; N]) -> Self {
            let mut res = Self {
                item_set: HashSet::new(),
            };
            for item in items {
                res.item_set.insert(item);
            }
            res
        }
    }

    impl ItemSet {
        pub fn new() -> Self {
            Self {
                item_set: HashSet::new(),
            }
        }

        pub fn insert(&mut self, value: LR0Item) -> bool {
            self.item_set.insert(value)
        }

        pub fn iter(&self) -> impl Iterator<Item = &LR0Item> + '_ {
            self.item_set.iter()
        }

        pub fn closure(&self, prod_list: &Vec<Production>) -> Self {
            let mut pre = self.clone();
            loop {
                let mut new = pre.clone();
                for item in pre.iter() {
                    let next_symbol = item.peek();
                    if let Some(GrammarSymbol::NonTerminalSymbol(inner)) = next_symbol {
                        let left = GrammarSymbol::NonTerminalSymbol(inner.clone());
                        for prod in prod_list.iter() {
                            if prod.left == left {
                                new.item_set.insert(LR0Item::new(prod.clone(), 0));
                            }
                        }
                    }
                }
                if new == pre {
                    return new;
                } else {
                    pre = new;
                }
            }
        }
    }

    #[derive(Debug)]
    pub struct TransitionEdge {
        driver: GrammarSymbol,
        from: usize,
        to: usize,
    }

    impl TransitionEdge {
        pub fn new(from: usize, to: usize, driver: GrammarSymbol) -> Self {
            Self { from, to, driver }
        }
    }

    pub struct DFA {
        pub grammar: Grammar,
        pub states: Vec<ItemSet>,
        pub start: usize,
        pub edges: Vec<TransitionEdge>,
    }

    impl Debug for DFA {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            writeln!(f, "states:")?;
            writeln!(f, "{:?}", self.states)?;
            writeln!(f, "edges:")?;
            writeln!(f, "{:?}", self.edges)
        }
    }

    impl DFA {
        /// Build DFA from the start item set S0
        /// The S0 will do closure automatically
        /// # Params
        /// + 's0': start state
        /// + 'grammar': grammar of the dfa
        pub fn from(s0: ItemSet, grammar: Grammar) -> Self {
            Self {
                grammar: grammar.clone(),
                states: vec![s0.closure(&grammar.productions)],
                start: 0,
                edges: vec![],
            }
        }

        /// include
        /// 1. iterator driver
        /// 2. build following item set
        /// 3. judge if the following set is new
        pub fn exhaust_transition(&mut self, from: usize) {
            let mut next_map = HashMap::new();
            for item in self.states[from].item_set.iter() {
                if let Some(driver) = item.peek() {
                    if !next_map.contains_key(driver) {
                        next_map.insert(driver.clone(), ItemSet::new());
                    }
                    let mut item = item.clone();
                    item.next();
                    next_map.get_mut(driver).unwrap().item_set.insert(item);
                }
            }
            for (k, v) in next_map.iter() {
                let v = v.closure(&self.grammar.productions);
                if self.states.contains(&v) {
                    continue;
                }
                let v = v.clone();
                self.states.push(v);
                self.edges
                    .push(TransitionEdge::new(from, self.states.len() - 1, k.clone()));
            }
        }

        // Build complete DFA
        pub fn build(&mut self) {
            let mut cur = 0;
            while cur < self.states.len() {
                self.exhaust_transition(cur);
                cur += 1;
            }
        }

        // a SLR(1) should be like
        // check the X->a·bg item
        // If for any Y->a·, b in FOLLOW(Y), then is not
        pub fn is_slr1(&mut self) -> bool {
            for item_set in &self.states {
                let shifts: Vec<_> = item_set.iter().filter(|i| i.is_shift()).collect();
                let reductions: Vec<_> = item_set.iter().filter(|i| i.is_reduction()).collect();
                for shift in shifts.iter() {
                    let peek = shift.peek().unwrap();
                    for reduction in reductions.iter() {
                        let reduction_follow = self.grammar.follow(reduction.production.left.clone());
                        if reduction_follow.contains(peek) {
                            return false;
                        }
                    }
                }
            }
            true
        }

        pub fn to_parse_table(&self) -> (Vec<ActionCell>, Vec<GoToCell>, Vec<Production>) {
            let mut action_table = vec![];
            let mut goto_table = vec![];
            let mut production2id = HashMap::new();
            let mut production_cnt = 0;
            let mut map: HashMap<usize, HashMap<GrammarSymbol, usize>> = HashMap::new();
            for i in 0..self.states.len() {
                map.insert(i, HashMap::new());
            }
            for edge in &self.edges {
                map.get_mut(&edge.from).unwrap().insert(edge.driver.clone(), edge.to);
            }
            for i in 0..self.states.len() {
                let state = &self.states[i];
                for item in state.iter() {
                    if !production2id.contains_key(&item.production) {
                        production2id.insert(item.production.clone(), production_cnt);
                        production_cnt += 1;
                    }
                    if item.is_shift() {
                        let symbol = item.peek().unwrap().clone();
                        let next = map.get(&i).unwrap().get(&symbol).unwrap();
                        action_table.push(ActionCell::new(i, symbol, Action::Shift(*next)));
                    } else if item.is_reduction() {
                        let next = production2id.get(&item.production).unwrap();
                        for terminal in &self.grammar.terminal {
                            action_table.push(ActionCell::new(i, terminal.clone(), Action::Reduction(*next)))
                        }
                    } else if item.is_accepted() {
                        action_table.push(ActionCell::new(i, GrammarSymbol::End, Action::Accept));
                    }
                }
                for (symbol, next) in map.get(&i).unwrap() {
                    if symbol.is_non_terminal() {
                        goto_table.push(GoToCell::new(i, symbol.clone(), next.clone()));
                    }
                }
            }
            let mut productions = vec![];
            productions.resize(
                production2id.len(),
                Production::new(GrammarSymbol::Null, vec![GrammarSymbol::Null]),
            );
            for (p, i) in &production2id {
                productions[*i] = p.clone();
            }
            (action_table, goto_table, productions)
        }
    }

    #[derive(Debug)]
    pub struct ActionCell {
        state: usize,
        terminal: GrammarSymbol,
        action: Action,
    }

    impl ActionCell {
        fn new(state: usize, terminal: GrammarSymbol, action: Action) -> Self {
            Self {
                state,
                terminal,
                action,
            }
        }
    }

    #[derive(Debug)]
    pub enum Action {
        Reduction(usize),
        Shift(usize),
        Accept,
    }

    #[derive(Debug)]
    pub struct GoToCell {
        state: usize,
        non_terminal: GrammarSymbol,
        next: usize,
    }

    impl GoToCell {
        fn new(state: usize, non_terminal: GrammarSymbol, next: usize) -> Self {
            Self {
                state,
                non_terminal,
                next,
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::grammar::ll1::{is_ll1, to_ll1_parse_table};
    use crate::{grammar::lr0::DFA, symbol::GrammarSymbol};
    use std::collections::HashSet;

    use crate::symbol::Production;

    use super::{
        lr0::{ItemSet, LR0Item},
        Grammar,
    };

    // A->ab
    #[test]
    fn test_grammar_first_simple() {
        let mut grammar = Grammar::new(vec![Production::like('A', "ab")]);
        let output = grammar.first('A');
        assert_eq!(output.len(), 1);
        assert!(output.contains(&'a'.into()));
    }

    // A->ab
    // B->Ab
    #[test]
    fn test_grammar_first_non_terminal() {
        let mut grammar = Grammar::new(vec![Production::like('A', "ab"), Production::like('B', "Ab")]);
        let output = grammar.first('B');
        assert_eq!(output.len(), 1);
        assert!(output.contains(&'a'.into()));
    }

    // A->e
    // B->AAb
    // C->AA
    #[test]
    fn test_grammar_first_with_null() {
        let mut grammar = Grammar::new(vec![
            Production::new('A'.into(), vec![GrammarSymbol::Null]),
            Production::like('B', "AAb"),
            Production::like('C', "AA"),
        ]);
        let b_first = grammar.first('B');
        assert_eq!(b_first.len(), 1);
        assert!(b_first.contains(&'b'.into()));
        let c_first = grammar.first('C');
        assert_eq!(c_first.len(), 1);
        assert!(c_first.contains(&GrammarSymbol::Null));
    }

    // A->aBc
    // B->b
    #[test]
    fn test_grammar_follow() {
        let mut grammar = Grammar::new(vec![Production::like('A', "aBc"), Production::like('B', "b")]);
        let output = grammar.follow('B');
        assert_eq!(output.len(), 1);
        assert!(output.contains(&'c'.into()));
    }

    // 1. A->ab, A->bc, B->Aa
    // 2. A->ab, A->ac
    #[test]
    fn test_is_ll1() {
        let mut grammar = Grammar::new(vec![
            Production::like('A', "ab"),
            Production::like('A', "bc"),
            Production::like('B', "Aa"),
        ]);
        assert_eq!(is_ll1(&mut grammar), true);
    }

    #[test]
    fn test_to_ll1_table() {
        let mut grammar = Grammar::new(vec![
            Production::like('A', "ab"),
            Production::like('A', "bc"),
            Production::like('B', "Aa"),
        ]);
        println!("{:?}", to_ll1_parse_table(&mut grammar));
    }

    // S = {A->·B}
    // additional: B->b
    // closure should be {A->·B, B->·b}
    #[test]
    fn test_lr0item_closure() {
        let prod1 = Production::like('A', "B");
        let prod2 = Production::like('B', "b");
        let prod_set = vec![prod1.clone(), prod2.clone()];
        let item_set = ItemSet::from([LR0Item::like('A', "B", 0)]);
        let closure = item_set.closure(&prod_set);
        assert_eq!(
            closure.item_set,
            HashSet::from([LR0Item::new(prod1, 0), LR0Item::new(prod2, 0)])
        );
    }

    // A->Ba
    #[test]
    fn test_simple_translation() {
        let grammar = Grammar::new(vec![Production::like('A', "Ba")]);
        let s0 = ItemSet::from([LR0Item::like('A', "Ba", 0)]);
        let mut dfa = DFA::from(s0, grammar);
        dfa.exhaust_transition(0);
        dfa.exhaust_transition(1);

        let target = vec![
            ItemSet::from([LR0Item::new(Production::like('A', "Ba"), 0)]),
            ItemSet::from([LR0Item::new(Production::like('A', "Ba"), 1)]),
            ItemSet::from([LR0Item::new(Production::like('A', "Ba"), 2)]),
        ];
        assert_eq!(target.len(), 3);
        for x in target {
            assert!(dfa.states.contains(&x));
        }
    }

    // {S'->S, S->aA, A->a}
    // S0 = {S'->·S, S->·aA}
    // S1 should be GO(S0, S) = {S'->S·}
    // S2 should be GO(S0, a) = {S->a·A, A->·a}
    // S3 should be GO(S2, A) = {S->aA·}
    // S4 should be GO(s2, a) = {A->a·}
    #[test]
    fn test_build() {
        let start: GrammarSymbol = 'S'.into();
        let start_dash = start.dash();
        let grammar = Grammar::new(vec![
            Production::new(start_dash.clone(), vec![start.clone()]),
            Production::like('S', "aA"),
            Production::like('A', "a"),
        ]);
        let s0 = ItemSet::from([LR0Item::new(
            Production::new(start_dash.clone(), vec![start.clone()]),
            0,
        )]);
        let mut dfa = DFA::from(s0, grammar);
        dfa.build();

        let targets = vec![
            ItemSet::from([
                LR0Item::new(Production::new(start_dash.clone(), vec![start.clone()]), 0),
                LR0Item::like('S', "aA", 0),
            ]),
            ItemSet::from([LR0Item::new(
                Production::new(start_dash.clone(), vec![start.clone()]),
                1,
            )]),
            ItemSet::from([LR0Item::like('S', "aA", 1), LR0Item::like('A', "a", 0)]),
            ItemSet::from([LR0Item::like('S', "aA", 2)]),
            ItemSet::from([LR0Item::like('A', "a", 1)]),
        ];
        assert_eq!(targets.len(), dfa.states.len());
        for target in targets {
            assert!(dfa.states.contains(&target));
        }
    }

    // {S'->S, S->aA, A->a}
    // S0 = {S'->·S, S->·aA}
    // S1 should be GO(S0, S) = {S'->S·}
    // S2 should be GO(S0, a) = {S->a·A, A->·a}
    // S3 should be GO(S2, A) = {S->aA·}
    // S4 should be GO(s2, a) = {A->a·}
    #[test]
    fn test_to_lr0_table() {
        let start: GrammarSymbol = 'S'.into();
        let start_dash = start.dash();
        let grammar = Grammar::new(vec![
            Production::new(start_dash.clone(), vec![start.clone()]),
            Production::like('S', "aA"),
            Production::like('A', "a"),
        ]);
        let s0 = ItemSet::from([LR0Item::new(
            Production::new(start_dash.clone(), vec![start.clone()]),
            0,
        )]);
        let mut dfa = DFA::from(s0, grammar);
        dfa.build();
        println!("{:?}", dfa);
        let (action, goto, productions) = dfa.to_parse_table();
        println!("{:?}\n{:?}\n{:?}", action, goto, productions);
    }
}
