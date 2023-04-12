use crate::symbol::{FirstSet, FollowSet, GrammarSymbol, Production};
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
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
            productions: productions,
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

    pub fn follow(&mut self, symbol: impl Into<GrammarSymbol>) -> FollowSet {
        let symbol = symbol.into();
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
                                self.follow_map
                                    .insert(production.left.clone(), follow_left.clone());
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

pub mod LL1 {
    use crate::symbol::{GrammarSymbol, Production};

    // Cell of LL Parsing Table
    pub struct Cell {
        stack_top: GrammarSymbol,
        input: GrammarSymbol,
        production: Production,
    }
}

pub mod LR0 {
    use std::{
        collections::{HashMap, HashSet},
        hash::Hash,
    };

    use crate::symbol::{FollowSet, GrammarSymbol, Production};

    #[derive(Debug, PartialEq, Eq, Hash, Clone)]
    pub struct LR0Item {
        production: Production,
        dot_pos: usize,
    }

    impl LR0Item {
        pub fn new(production: Production) -> Self {
            Self {
                production,
                dot_pos: 0,
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
            self.dot_pos < self.production.right.len()
                && self.production.right[self.dot_pos].is_terminal()
        }

        pub fn is_reduction(&self) -> bool {
            self.dot_pos == self.production.right.len() && !self.production.left.is_start()
        }
    }

    #[derive(Debug, Eq, PartialEq, Clone)]
    pub struct ItemSet {
        pub item_set: HashSet<LR0Item>,
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

        pub fn closure(&self, prod_list: Vec<Production>) -> Self {
            let mut pre = self.clone();
            loop {
                let mut new = pre.clone();
                for item in pre.iter() {
                    let next_symbol = item.peek();
                    if let Some(GrammarSymbol::NonTerminalSymbol(inner)) = next_symbol {
                        let left = GrammarSymbol::NonTerminalSymbol(inner.clone());
                        for prod in prod_list.iter() {
                            if prod.left == left {
                                new.item_set.insert(LR0Item::new(prod.clone()));
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

    #[derive(Debug)]
    pub struct DFA {
        states: Vec<ItemSet>,
        start: usize,
        edges: Vec<TransitionEdge>,
    }

    impl DFA {
        // Build DFA from the start item set S0
        pub fn from(s0: ItemSet) -> Self {
            Self {
                states: vec![s0],
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
                if self.states.contains(v) {
                    continue;
                }
                let mut v = v.clone();
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
        pub fn is_SLR1(&self, follow_map: HashMap<GrammarSymbol, FollowSet>) -> bool {
            for item_set in &self.states {
                let shifts: Vec<_> = item_set.iter().filter(|i| i.is_shift()).collect();
                let reductions: Vec<_> = item_set.iter().filter(|i| i.is_reduction()).collect();
                for shift in shifts.iter() {
                    let peek = shift.peek().unwrap();
                    for reduction in reductions.iter() {
                        let reduction_follow = follow_map.get(&reduction.production.left).unwrap();
                        if reduction_follow.contains(peek) {
                            return false;
                        }
                    }
                }
            }
            true
        }

        // pub fn to_parse_table(&self) -> (Vec<ActionCell>, Vec<GoToCell>, Vec<Production>) {
        //     let production2id:  = HashMap::new();
        //     for i in 0..self.states.len() {
        //         let state = &self.states[i];
        //         for item in state.iter() {
        //
        //         }
        //     }
        //     todo!()
        // }
    }

    pub struct ActionCell {
        state: usize,
        terminal: GrammarSymbol,
        action: Action,
    }

    pub enum Action {
        Reduction(usize),
        Shift(usize),
        Accept,
    }

    pub struct GoToCell {
        state: usize,
        non_terminal: GrammarSymbol,
        next: usize,
    }
}

#[cfg(test)]
mod test {
    use crate::{grammar::LR0::DFA, symbol::GrammarSymbol};
    use std::collections::HashSet;

    use crate::symbol::Production;

    use super::{
        Grammar,
        LR0::{ItemSet, LR0Item},
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
    fn test_grammar_first_nonterminal() {
        let mut grammar = Grammar::new(vec![
            Production::like('A', "ab"),
            Production::like('B', "Ab"),
        ]);
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
        let mut grammar = Grammar::new(vec![
            Production::like('A', "aBc"),
            Production::like('B', "b"),
        ]);
        let output = grammar.follow('B');
        assert_eq!(output.len(), 1);
        assert!(output.contains(&'c'.into()));
    }

    // S = {A->·B}
    // additional: B->b
    // closure should be {A->·B, B->·b}
    #[test]
    fn test_LR0Item_closure() {
        let prod1 = Production::like('A', "B");
        let prod2 = Production::like('B', "b");
        let prod_set = vec![prod1.clone(), prod2.clone()];
        let mut item_set = ItemSet::new();
        item_set.insert(LR0Item::new(prod1.clone()));
        let closure = item_set.closure(prod_set);
        assert_eq!(
            closure.item_set,
            HashSet::from([LR0Item::new(prod1), LR0Item::new(prod2)])
        );
    }

    // S0 = {A->·Ba}
    // S1 should be {A->B·a}
    // S2 should be {A->Ba·}
    #[test]
    fn test_simple_translation() {
        let mut s0 = ItemSet::new();
        s0.insert(LR0Item::new(Production::like('A', "Ba")));
        let mut dfa = DFA::from(s0);
        dfa.exhaust_transition(0);
        dfa.exhaust_transition(1);
        println!("{:?}", dfa);
    }

    // S0 = {A->·Ba}
    // complete DFA should be
    // {{A->·Ba}, {A->B·a}, {A->Ba·}}
    #[test]
    fn test_build() {
        let mut s0 = ItemSet::new();
        s0.insert(LR0Item::new(Production::like('A', "Ba")));
        let mut dfa = DFA::from(s0);
        dfa.build();
        println!("{:?}", dfa);
    }
}
