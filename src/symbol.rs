use crate::graph::LexemeCategory;
use std::collections::{HashMap, HashSet};

pub type FirstSet = HashSet<GrammarSymbol>;
pub type FollowSet = HashSet<GrammarSymbol>;

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub enum GrammarSymbol {
    Null, // e symbol
    End,  // # symbol
    TerminalSymbol(String),
    NonTerminalSymbol(String),
}

impl GrammarSymbol {
    // default: upper class letter is NonTerminal, other is Terminal
    pub fn new(ch: char) -> GrammarSymbol {
        match ch {
            'A'..='Z' => GrammarSymbol::NonTerminalSymbol(ch.to_string()),
            _ => GrammarSymbol::TerminalSymbol(ch.to_string()),
        }
    }

    /// Generate GrammarSymbol string from String
    /// ```rust
    /// use compiler_expt::symbol;
    /// use compiler_expt::symbol::GrammarSymbol;
    /// let x = GrammarSymbol::chain("abcd");
    /// let y = vec![symbol!('a'), symbol!('b'), symbol!('c'), symbol!('d')];
    /// assert_eq!(x, y);
    /// ```
    pub fn chain(s: &str) -> Vec<GrammarSymbol> {
        let mut res = vec![];
        for ch in s.chars() {
            res.push(GrammarSymbol::new(ch))
        }
        res
    }

    pub fn dash(&self) -> GrammarSymbol {
        match self {
            GrammarSymbol::NonTerminalSymbol(inner) => {
                let new_inner = inner.clone() + "'";
                GrammarSymbol::NonTerminalSymbol(new_inner)
            }
            _ => panic!("method GrammarSymbol.sub() is only applied to NonTerminalSymbol"),
        }
    }

    /// Get first_set from first map
    /// Do not calculate first_set
    /// ```rust
    /// use compiler_expt::symbol::GrammarSymbol;
    /// use std::collections::{HashSet, HashMap};
    ///
    /// let fs = GrammarSymbol::get_first('a'.into(), &HashMap::new());
    /// assert_eq!(fs, HashSet::from(['a'.into()]));
    /// ```
    pub fn get_first(symbol: GrammarSymbol, first_map: &HashMap<GrammarSymbol, FirstSet>) -> FirstSet {
        match symbol {
            GrammarSymbol::NonTerminalSymbol(_) => first_map.get(&symbol).unwrap().clone(),
            other => HashSet::from([other]),
        }
    }

    // define 'S' as the start state
    pub fn is_start(&self) -> bool {
        match self {
            GrammarSymbol::NonTerminalSymbol(content) => content == "S",
            _ => false,
        }
    }

    pub fn is_terminal(&self) -> bool {
        match self {
            GrammarSymbol::TerminalSymbol(_) => true,
            _ => false,
        }
    }

    pub fn is_non_terminal(&self) -> bool {
        match self {
            GrammarSymbol::NonTerminalSymbol(_) => true,
            _ => false,
        }
    }
}

impl From<char> for GrammarSymbol {
    fn from(value: char) -> Self {
        if value >= 'A' && value <= 'Z' {
            GrammarSymbol::NonTerminalSymbol(value.to_string())
        } else {
            GrammarSymbol::TerminalSymbol(value.to_string())
        }
    }
}

impl From<&str> for GrammarSymbol {
    fn from(value: &str) -> Self {
        let head = value.chars().nth(0).unwrap();
        if value.len() == 1 {
            head.into()
        } else if head >= 'A' && head <= 'Z' {
            GrammarSymbol::NonTerminalSymbol(value.to_string())
        } else {
            GrammarSymbol::TerminalSymbol(value.to_string())
        }
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
// The production will pre-build FIRST set
pub struct Production {
    pub left: GrammarSymbol,
    pub right: Vec<GrammarSymbol>,
}

impl Production {
    pub fn new(left: GrammarSymbol, right: Vec<GrammarSymbol>) -> Self {
        Self { left, right }
    }

    pub fn like(left: char, right: &str) -> Self {
        Self {
            left: GrammarSymbol::new(left),
            right: GrammarSymbol::chain(right),
        }
    }

    pub fn right_nth(&self, index: usize) -> Option<&GrammarSymbol> {
        self.right.get(index)
    }

    pub fn first(&self, first_map: &HashMap<GrammarSymbol, FirstSet>) -> FirstSet {
        let mut res = FirstSet::new();
        if self.is_null() {
            res.insert(GrammarSymbol::Null);
            return res;
        }
        let mut k = 0;
        let n = self.right.len();
        while k < n {
            let first_x_k = GrammarSymbol::get_first(self.right[k].clone(), first_map);
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

    pub fn find(&self, s: &GrammarSymbol) -> Option<usize> {
        for i in 0..self.right.len() {
            if self.right[i] == *s {
                return Some(i);
            }
        }
        return None;
    }

    pub fn is_null(&self) -> bool {
        if self.right.len() != 1 {
            return false;
        }
        match self.right[0] {
            GrammarSymbol::Null => true,
            _ => false,
        }
    }
}

// A-> Aa1 | Aa2 | ... | Aan | b1 | b2 | ... bn
// make sure that the 'left' part in productions are the same
pub fn has_left_recursion(productions: &Vec<Production>) -> bool {
    let left = &productions[0].left;
    for production in productions {
        if production.right.len() >= 1 && production.right[0] == *left {
            return true;
        }
    }
    return false;
}

// A-> Aa1 | Aa2 | ... | Aan | b1 | b2 | ... bn
// into
// A-> b1A'|b2A'|...|bnA'
// A'->a1A'|a2A'|...|anA'|e
pub fn eliminate_left_recursion(productions: &Vec<Production>) -> Vec<Production> {
    let left = &productions[0].left;
    let left_dash = left.dash();

    let mut res = vec![Production::new(left_dash.clone(), vec![GrammarSymbol::Null])];
    for production in productions {
        if production.right.len() >= 1 && production.right[0] == *left {
            // is recursive
            let mut new_right_inner = Vec::from(&production.right[1..]);
            new_right_inner.push(left_dash.clone());
            res.push(Production::new(left_dash.clone(), new_right_inner));
        } else {
            // is not recursive
            let mut new_right_inner = production.right.clone();
            new_right_inner.push(left_dash.clone());
            res.push(Production::new(left.clone(), new_right_inner));
        }
    }
    res
}

// test if the productions has common factor
// return the common factor or None
// if the productions has multiple common factors, only return one
pub fn has_common_factor(productions: &Vec<Production>) -> Option<GrammarSymbol> {
    let mut cnt = HashSet::new();
    for production in productions {
        if cnt.contains(&production.right[0]) {
            return Some(production.right[0].clone());
        }
        cnt.insert(production.right[0].clone());
    }
    None
}

// extract the specified factor
pub fn extract_common_factor(productions: &Vec<Production>, factor: GrammarSymbol) -> Vec<Production> {
    let left = &productions[0].left;
    let left_dash = left.dash();
    let mut res = vec![Production::new(left.clone(), vec![factor.clone(), left_dash.clone()])];
    for production in productions {
        if production.right[0] == factor {
            let new_right_inner = Vec::from(&production.right[1..]);
            res.push(Production::new(left_dash.clone(), new_right_inner));
        } else {
            res.push(Production::new(left.clone(), production.right.clone()));
        }
    }
    res
}

// function that returns the FIRST set of symbol
// the symbol can be any type of GrammarSymbol
pub fn calc_first(symbol: GrammarSymbol, production_set: &HashSet<Production>, first_map: &HashMap<GrammarSymbol, FirstSet>) -> FirstSet {
    match symbol {
        GrammarSymbol::NonTerminalSymbol(ref String) => {
            let productions: Vec<&Production> = production_set.iter().filter(|&prod| prod.left == symbol).collect();
            let mut first_set = FirstSet::new();
            for p in productions {
                if p.is_null() {
                    first_set.insert(GrammarSymbol::Null);
                    continue;
                }
                first_set.extend(p.first(first_map));
            }
            first_set
        }
        other => HashSet::from([other]),
    }
}

// function that return the FOLLOW set
// requires FIRST set
// the function may add more FOLLOW set if necessary
pub fn calc_follow(
    target: GrammarSymbol,
    production_set: &HashSet<Production>,
    first_map: &HashMap<GrammarSymbol, FirstSet>,
    follow_map: &mut HashMap<GrammarSymbol, FollowSet>,
) -> FollowSet {
    let mut res = FollowSet::new();
    for production in production_set {
        if let Some(pos) = production.find(&target) {
            if pos == production.right.len() - 1 {
                if let Some(follow_a) = follow_map.get(&production.left) {
                    res.extend(follow_a.clone());
                } else {
                    let follow_a = calc_follow(production.left.clone(), production_set, first_map, follow_map);
                    follow_map.insert(production.left.clone(), follow_a.clone());
                    res.extend(follow_a);
                }
            } else {
                let mut first_beta = GrammarSymbol::get_first(production.right[pos + 1].clone(), first_map);
                first_beta.remove(&GrammarSymbol::Null);
                res.extend(first_beta);
                // 先跳过beta推导出epsilon的情况
            }
        }
    }
    res
}

#[cfg(test)]
mod test {
    use crate::symbol::{
        calc_first, eliminate_left_recursion, extract_common_factor, has_common_factor, has_left_recursion, FirstSet, GrammarSymbol,
        Production,
    };
    use std::{
        collections::{HashMap, HashSet},
        hash::Hash,
    };

    use super::calc_follow;

    // A->Ab | c
    #[test]
    fn test_has_left_recursion() {
        let productions = vec![Production::like('A', "Ab"), Production::like('A', "c")];
        assert_eq!(has_left_recursion(&productions), true);
    }

    // A->Ab | c
    // ---
    // A->cA'
    // A'->bA'|e
    #[test]
    fn test_eliminate_left_recursion() {
        let productions = vec![Production::like('A', "Ab"), Production::like('A', "c")];
        let new_productions = eliminate_left_recursion(&productions);
        let left: GrammarSymbol = 'A'.into();
        let left_dash = left.dash();
        let target = vec![
            Production::new(left_dash.clone(), vec![GrammarSymbol::Null]),
            Production::new(left.clone(), vec!['c'.into(), left_dash.clone()]),
            Production::new(left_dash.clone(), vec!['b'.into(), left_dash.clone()]),
        ];
        assert_eq!(new_productions.len(), target.len());
        for p in target {
            assert!(new_productions.contains(&p));
        }
    }

    // A->ab|ac|d
    #[test]
    fn test_has_common_factor() {
        let productions = vec![Production::like('A', "ab"), Production::like('A', "ac"), Production::like('A', "d")];
        assert_eq!(has_common_factor(&productions), Some('a'.into()));
    }

    // A->ab|ac|d
    // ---
    // A->aA'|d
    // A'->b|c
    #[test]
    fn test_extract_common_factor() {
        let productions = vec![Production::like('A', "ab"), Production::like('A', "ac"), Production::like('A', "d")];
        let new_productions = extract_common_factor(&productions, 'a'.into());
        let left: GrammarSymbol = 'A'.into();
        let left_dash = left.dash();
        let target = vec![
            Production::like('A', "d"),
            Production::new('A'.into(), vec!['a'.into(), left_dash.clone()]),
            Production::new(left_dash.clone(), vec!['b'.into()]),
            Production::new(left_dash.clone(), vec!['c'.into()]),
        ];
        println!("{:?}", &new_productions);
        assert_eq!(new_productions.len(), target.len());
        for p in target {
            assert!(new_productions.contains(&p));
        }
    }

    // A->ab
    #[test]
    fn test_first_simple() {
        let left = 'A'.into();
        let right = GrammarSymbol::chain("ab");
        let mut first_map = HashMap::new();
        let prod = Production::new(left, right);
        assert_eq!(prod.first(&first_map), HashSet::from(['a'.into()]));
    }

    // A->ab
    // B->Ab
    #[test]
    fn test_first_non_terminal() {
        let mut first_map = HashMap::new();
        let prod_set = HashSet::from([Production::like('A', "ab")]);
        let first = calc_first('A'.into(), &prod_set, &first_map);
        first_map.insert('A'.into(), first);

        let left2 = 'B'.into();
        let right2 = GrammarSymbol::chain("Ab");
        let prod = Production::new(left2, right2);
        assert_eq!(prod.first(&first_map), HashSet::from(['a'.into()]));
    }

    // A -> epsilon
    // B-> AAb
    #[test]
    fn test_first_with_e() {
        let left1: GrammarSymbol = 'A'.into();
        let right1 = vec![GrammarSymbol::Null];
        let mut first_map = HashMap::new();
        let prod_set = HashSet::from([Production::new(left1.clone(), right1)]);
        let first = calc_first(left1.clone(), &prod_set, &first_map);
        assert_eq!(first.clone(), HashSet::from([GrammarSymbol::Null]));
        first_map.insert(left1.clone(), first);

        let left2 = 'B'.into();
        let right2 = GrammarSymbol::chain("AAb");
        let prod = Production::new(left2, right2);
        assert_eq!(prod.first(&first_map), HashSet::from(['b'.into()]));
    }

    // A->aBc
    #[test]
    fn test_follow() {
        let left: GrammarSymbol = 'A'.into();
        let right = GrammarSymbol::chain("aBc");
        let prod = Production::new(left, right);
        let prod_set = HashSet::from([prod]);
        let first_map = HashMap::new();
        let mut follow_map = HashMap::new();
        let follow = calc_follow('B'.into(), &prod_set, &first_map, &mut follow_map);
        assert_eq!(follow, HashSet::from(['c'.into()]));
    }
}
