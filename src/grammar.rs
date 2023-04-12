use std::collections::HashMap;
use crate::symbol::{FirstSet, FollowSet, GrammarSymbol, Production};

pub struct Grammar {
    terminal: Vec<GrammarSymbol>,
    productions: Vec<Production>,
    first_map: HashMap<GrammarSymbol, FirstSet>,
    follow_map: HashMap<GrammarSymbol, FollowSet>
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
            Self { production, dot_pos: 0 }
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

    impl ItemSet {
        pub fn new() -> Self {
            Self { item_set: HashSet::new() }
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
                self.edges.push(TransitionEdge::new(from, self.states.len() - 1, k.clone()));
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
    use crate::grammar::LR0::DFA;
    use crate::graph::product;
    use std::collections::HashSet;

    use crate::symbol::Production;

    use super::LR0::{ItemSet, LR0Item};

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
        assert_eq!(closure.item_set, HashSet::from([LR0Item::new(prod1), LR0Item::new(prod2)]));
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
