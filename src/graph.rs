use crate::charset::{charsets_in, CharSetTable};

use queues::{queue, IsQueue, Queue};
use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::iter::Iterator;
use std::rc::Rc;
use std::slice::Iter;

const NULL_DRIVER_ID: usize = 0xFFFFFFFF;

#[derive(Copy, Clone, Debug)]
pub enum LexemeCategory {
    Nil,             // not a lexeme category
    IntegerConst,    //
    FloatConst,      //
    ScientificConst, //
    NumericConst,    //
    Note,            //
    StringConst,     //
    SpaceConst,      //
    CompareOperator, //
    Ident,           //
    LogicOperator,   //
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DriverType {
    Null,
    Char,
    Charset,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StateType {
    Match,
    Unmatch,
}

// basic graph
// represent both nfa and dfa
// first state is the start state
// if is nfa, last state is the end state
#[derive(Clone, Debug)]
pub struct Graph {
    graph_id: usize,
    edge_table: Vec<Edge>,
    state_table: Vec<State>,
    charset_table: Rc<CharSetTable>,
}

impl Display for Graph {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "graph_id: {}", self.graph_id)?;
        writeln!(f, "edge_table:")?;
        for edge in &self.edge_table {
            writeln!(f, "{:?}", edge)?;
        }
        writeln!(f, "state_table:")?;
        for state in &self.state_table {
            writeln!(f, "{:?}", state)?;
        }
        writeln!(f, "charset_table:")?;
        for charset in self.charset_table.iter() {
            writeln!(f, "{:?}", charset)?;
        }
        Ok(())
    }
}

impl Graph {
    pub fn new(edge_table: Vec<Edge>, state_table: Vec<State>, charset_table: &Rc<CharSetTable>) -> Self {
        Graph {
            graph_id: 0,
            edge_table,
            state_table,
            charset_table: Rc::clone(charset_table),
        }
    }

    pub fn edges(&self) -> Iter<Edge> {
        self.edge_table.iter()
    }

    pub fn states(&self) -> Iter<State> {
        self.state_table.iter()
    }

    pub fn num_of_states(&self) -> usize {
        self.state_table.len()
    }

    pub fn end_state(&mut self) -> &mut State {
        let num_of_states = self.num_of_states();
        &mut self.state_table[num_of_states - 1]
    }

    pub fn add_state(&mut self, state: State) {
        self.state_table.push(state)
    }

    pub fn append_states(&mut self, mut states: Vec<State>) {
        self.state_table.append(&mut states);
    }

    pub fn add_edge(&mut self, edge: Edge) {
        self.edge_table.push(edge)
    }

    pub fn append_edges(&mut self, mut edges: Vec<Edge>) {
        self.edge_table.append(&mut edges);
    }

    pub fn get_state_by_id(&self, id: usize) -> Option<&State> {
        self.states().find(|s| s.state_id == id)
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    from_state: usize,
    next_state: usize,
    driver_id: usize,
    driver_type: DriverType,
}

impl Edge {
    pub fn new(from_state: usize, next_state: usize, driver_id: usize, driver_type: DriverType) -> Self {
        Edge {
            from_state,
            next_state,
            driver_id,
            driver_type,
        }
    }
}

#[derive(Clone, Debug)]
pub struct State {
    state_id: usize,
    state_type: StateType,
    category: LexemeCategory,
}

impl State {
    fn new(state_id: usize) -> Self {
        Self {
            state_id,
            state_type: StateType::Unmatch,
            category: LexemeCategory::Nil,
        }
    }

    fn set_type(mut self, state_type: StateType) -> Self {
        self.state_type = state_type;
        return self;
    }
}

// if the state has out edge
fn has_out_edge(state_id: usize, graph: &Graph) -> bool {
    for edge in graph.edges() {
        if edge.from_state == state_id {
            return true;
        }
    }
    return false;
}

// if the state has in edge
fn has_in_edge(state_id: usize, graph: &Graph) -> bool {
    for edge in graph.edges() {
        if edge.next_state == state_id {
            return true;
        }
    }
    return false;
}

/***********************
 Part1: basic operations
************************/

// add a null edge to the front of graph
fn push_null_front(nfa: &mut Graph) {
    // move states back
    nfa.state_table.iter_mut().for_each(|s| s.state_id = s.state_id + 1);
    nfa.edge_table.iter_mut().for_each(|e| {
        e.from_state = e.from_state + 1;
        e.next_state = e.next_state + 1;
    });
    // add new state and edge
    nfa.state_table.insert(0, State::new(0));
    nfa.add_edge(Edge::new(0, 1, NULL_DRIVER_ID, DriverType::Null));
}

// add a null edge to the back of graph
fn push_null_back(nfa: &mut Graph) {
    // set last state unmatch
    nfa.end_state().state_type = StateType::Unmatch;
    // add new state
    nfa.add_state(State::new(nfa.num_of_states()).set_type(StateType::Match));
    nfa.add_edge(Edge::new(
        nfa.num_of_states() - 2,
        nfa.num_of_states() - 1,
        NULL_DRIVER_ID,
        DriverType::Null,
    ));
}

pub fn generate_basic_nfa(driver_type: DriverType, driver_id: usize, table: &Rc<CharSetTable>) -> Graph {
    Graph::new(
        vec![Edge::new(0, 1, driver_id, driver_type)],
        vec![State::new(0), State::new(1).set_type(StateType::Match)],
        table,
    )
}

pub fn union_preprocess(nfa: &Graph) -> Graph {
    let mut new_nfa = nfa.clone();
    if has_in_edge(0, &new_nfa) {
        push_null_front(&mut new_nfa);
    }
    if has_out_edge(new_nfa.num_of_states() - 1, &new_nfa) {
        push_null_back(&mut new_nfa);
    }
    new_nfa
}

// union operation: nfa1 | nfa2
pub fn union(nfa1: &Graph, nfa2: &Graph) -> Graph {
    let nfa1 = union_preprocess(nfa1);
    let nfa2 = union_preprocess(nfa2);
    // s is the state number of nfa1 (except end state)
    let s = nfa1.num_of_states() - 1;
    // t is the state number of nfa2 (except end state)
    let t = nfa2.num_of_states() - 1;
    let mut new_nfa = nfa1.clone();
    new_nfa.end_state().state_id = s + t - 1;
    // change the identifier of end state from s to s+t-1
    new_nfa.edge_table.iter_mut().for_each(|edge| {
        edge.next_state = if edge.next_state == s { s + t - 1 } else { edge.next_state };
    });
    // add all states except start and end to new graph
    // all the identifiers should be plus s+1
    new_nfa.append_states(
        nfa2.states()
            .filter(|state| state.state_id > 0 && state.state_id < t - 1)
            .map(|state| State::new(state.state_id + s - 1))
            .collect(),
    );
    // process the state_id of each edge
    // if it's start state, remain 0
    // else, add up s+1
    new_nfa.append_edges(
        nfa2.edges()
            .map(|edge| {
                let from_state = if edge.from_state == 0 { 0 } else { edge.from_state + s - 1 };
                Edge::new(from_state, edge.next_state + s - 1, edge.driver_id, edge.driver_type)
            })
            .collect(),
    );
    new_nfa
}

// product operation: nfa1nfa2
pub fn product(nfa1: &Graph, nfa2: &Graph) -> Graph {
    let s = nfa1.num_of_states() - 1;
    // clone new_nfa from nfa1
    let mut new_nfa = nfa1.clone();
    // set the end_state to unmatch
    // if end of nfa1 has out edge or start of nfa2 has in edge, add null driver
    if has_out_edge(s, nfa1) && has_in_edge(0, nfa2) {
        push_null_back(&mut new_nfa);
    }
    new_nfa.end_state().state_type = StateType::Unmatch;
    // add all states of nfa2 except start state
    // add up s to state id
    new_nfa.append_states(
        nfa2.states()
            .filter(|state| state.state_id != 0)
            .map(|state| State::new(s + state.state_id))
            .collect(),
    );
    // set end state type to be match
    new_nfa.end_state().state_type = StateType::Match;
    // add edges
    new_nfa.append_edges(
        nfa2.edge_table
            .iter()
            .map(|edge| Edge::new(edge.from_state + s, edge.next_state + s, edge.driver_id, edge.driver_type))
            .collect(),
    );
    new_nfa
}

// plus-closure operation: nfa+
pub fn plus_closure(nfa: &Graph) -> Graph {
    let mut new_nfa = nfa.clone();
    // first add a null edge from raw end state to raw start state
    new_nfa.add_edge(Edge::new(new_nfa.num_of_states() - 1, 0, NULL_DRIVER_ID, DriverType::Null));
    // if start state has in edge
    // add new start state and null transformation
    // notice that I use raw nfa to do the judge
    if has_in_edge(0, &nfa) {
        push_null_front(&mut new_nfa);
    }
    // if end state has out edge
    // add new out state and null transformation
    if has_out_edge(nfa.num_of_states() - 1, &nfa) {
        push_null_back(&mut new_nfa);
    }
    new_nfa
}

// simple condition:
// 1. contains only two states
// 2. no edge to start state, no edge from end state
fn is_simple_for_closure(nfa: &Graph) -> bool {
    nfa.num_of_states() == 2 && !has_in_edge(0, nfa) && !has_out_edge(1, nfa)
}

// if the nfa is simple, the closure should be treated specially
fn simple_closure(nfa: &Graph) -> Graph {
    let mut new_nfa = Graph::new(vec![], vec![State::new(0).set_type(StateType::Match)], &nfa.charset_table);
    new_nfa.append_edges(nfa.edges().map(|edge| Edge::new(0, 0, edge.driver_id, edge.driver_type)).collect());
    new_nfa
}

// closure operation: nfa*
pub fn closure(nfa: &Graph) -> Graph {
    if is_simple_for_closure(nfa) {
        return simple_closure(nfa);
    }
    // reuse the code from plus_closure
    let mut new_nfa = plus_closure(&nfa);
    // add a null edge from start to end
    new_nfa.add_edge(Edge::new(0, new_nfa.num_of_states() - 1, NULL_DRIVER_ID, DriverType::Null));
    new_nfa
}

// zero-or-one operation: nfa?
pub fn zero_or_one(nfa: &Graph) -> Graph {
    let mut new_nfa = nfa.clone();
    // if start state has in edge
    // add new start state and null transformation
    if has_in_edge(0, &new_nfa) {
        push_null_front(&mut new_nfa);
    }
    // if end state has out edge
    // add new out state and null transformation
    if has_out_edge(new_nfa.num_of_states() - 1, &new_nfa) {
        push_null_back(&mut new_nfa);
    }
    // add a null edge from start to end
    new_nfa.add_edge(Edge::new(0, new_nfa.num_of_states() - 1, NULL_DRIVER_ID, DriverType::Null));
    new_nfa
}

/***********************
 Part2: transform to dfa
************************/

// definition of three usize:
// 1. state_id
// 2. symbol_id
// 3. state_id
// when state a meet symbol b, it transform to state c
pub type Dtran = HashMap<usize, HashMap<usize, usize>>;

#[derive(Debug, Clone)]
pub struct StateSet {
    inner: HashSet<usize>,
}

impl StateSet {
    fn new() -> Self {
        Self { inner: HashSet::new() }
    }

    fn from(v: Vec<usize>) -> Self {
        Self {
            inner: v.into_iter().collect(),
        }
    }

    fn contains(&self, value: &usize) -> bool {
        self.inner.contains(value)
    }

    fn insert(&mut self, value: usize) -> bool {
        self.inner.insert(value)
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Eq for StateSet {}

impl PartialEq for StateSet {
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        for value in &self.inner {
            if !other.inner.contains(value) {
                return false;
            }
        }
        return true;
    }
}

// return the e-closure of given state set
pub fn e_closure(from: &StateSet, nfa: &Graph) -> StateSet {
    let mut set = StateSet::clone(from);
    nfa.edges()
        .filter(|edge| from.contains(&edge.from_state) && edge.driver_type == DriverType::Null)
        .for_each(|edge| {
            set.insert(edge.next_state);
        });
    set
}

// return the state set which the given-state move to through given symbol
pub fn r#move(from: &StateSet, symbol: usize, nfa: &Graph) -> StateSet {
    let mut set = StateSet::new();
    nfa.edges()
        .filter(|edge| from.contains(&edge.from_state) && edge.driver_id == symbol)
        .for_each(|edge| {
            set.insert(edge.next_state);
        });
    set
}

// return two parts:
// 1. Dtran, all the elements is represented by usize
// 2. Matched States, stored in Vec
pub fn d_tran(nfa: &Graph) -> (Dtran, HashSet<usize>) {
    // return value
    let mut d_tran = Dtran::new();
    let mut matches = HashSet::new();
    // store the closure in vector, so the index is its final state_id
    let mut closures = vec![e_closure(&StateSet::from(vec![0]), nfa)];
    // add start closure to queue
    let mut queue = queue![0usize];
    // get set of symbol id
    let symbol_set = charsets_in(&nfa.charset_table);
    // start processing
    while let Ok(front) = queue.remove() {
        // check if it is matched state
        for state_id in &closures[front].inner {
            if nfa.get_state_by_id(*state_id).is_some_and(|st| st.state_type == StateType::Match) {
                matches.insert(front);
            }
        }
        let mut map = HashMap::new();
        // process transformation
        for symbol in &symbol_set {
            let next = e_closure(&r#move(&closures[front], *symbol, nfa), nfa);
            if next.is_empty() {
                continue;
            }
            if let Some(idx) = closures.iter().position(|v| *v == next) {
                // if is in the closures, only update d-tran
                map.insert(*symbol, idx);
            } else {
                closures.push(next);
                queue.add(closures.len() - 1).unwrap();
                map.insert(*symbol, closures.len() - 1);
            }
        }
        d_tran.insert(front, map);
    }
    (d_tran, matches)
}

// build dfa from a certain d-tran of a nfa
// the extra "matches" HashSet is used to mark matched state
pub fn nfa_to_dfa(nfa: &Graph) -> Graph {
    let mut dfa = Graph::new(vec![], vec![], &nfa.charset_table);
    let (d_tran, matches) = d_tran(nfa);
    for (k, v) in &d_tran {
        if matches.contains(k) {
            dfa.add_state(State::new(*k).set_type(StateType::Match));
        } else {
            dfa.add_state(State::new(*k));
        }
        for (s, n) in v.iter() {
            dfa.add_edge(Edge::new(*k, *n, *s, DriverType::Charset));
        }
    }
    dfa
}

#[cfg(test)]
mod test {
    use crate::charset::{CharSet, CharSetTable};
    use crate::graph::DriverType::Charset;
    use crate::graph::{
        closure, d_tran, e_closure, generate_basic_nfa, nfa_to_dfa, plus_closure, product, union, zero_or_one, DriverType, Edge, Graph,
        StateSet,
    };
    use std::rc::Rc;

    fn two_graph() -> (Graph, Graph) {
        let mut charset_table = CharSetTable::new();
        charset_table.push(CharSet::new(0, 0, 'a', 'a'));
        charset_table.push(CharSet::new(1, 0, 'b', 'b'));
        let charset_table = Rc::new(charset_table);
        let graph_a = generate_basic_nfa(Charset, 0, &charset_table);
        let graph_b = generate_basic_nfa(Charset, 1, &charset_table);
        (graph_a, graph_b)
    }

    fn one_graph() -> Graph {
        let mut charset_table = CharSetTable::new();
        charset_table.push(CharSet::new(0, 0, 'a', 'a'));
        let charset_table = Rc::new(charset_table);
        let graph = generate_basic_nfa(Charset, 0, &charset_table);
        graph
    }

    #[test]
    fn union_test() {
        let (graph_a, graph_b) = two_graph();
        let new_nfa = union(&graph_a, &graph_b);
        println!("{}", new_nfa);
    }

    #[test]
    fn product_test() {
        let (graph_a, graph_b) = two_graph();
        let new_nfa = product(&graph_a, &graph_b);
        println!("{}", new_nfa);
    }

    #[test]
    fn plus_closure_test() {
        let graph = one_graph();
        let new_nfa = plus_closure(&graph);
        println!("{}", new_nfa)
    }

    #[test]
    fn closure_test() {
        let graph = one_graph();
        let new_nfa = closure(&graph);
        println!("{}", new_nfa)
    }

    #[test]
    fn zero_or_one_test() {
        let graph = one_graph();
        let new_nfa = zero_or_one(&graph);
        println!("{}", new_nfa)
    }

    #[test]
    fn e_closure_test() {
        let (nfa1, nfa2) = two_graph();
        let mut nfa3 = product(&nfa1, &nfa2);
        println!("{}", nfa3);
        nfa3.edge_table.push(Edge::new(0, 2, 0, DriverType::Null));
        println!("{:?}", e_closure(&StateSet::from(vec![0]), &nfa3));
        println!("{:?}", e_closure(&StateSet::from(vec![0, 1]), &nfa3));
    }

    #[test]
    fn d_tran_test() {
        let (nfa1, nfa2) = two_graph();
        let nfa3 = product(&nfa1, &nfa2);
        println!("{}", nfa3);
        println!("{:?}", d_tran(&nfa3));
    }

    #[test]
    fn dfa_test() {
        let (nfa1, nfa2) = two_graph();
        let nfa3 = product(&nfa1, &nfa2);
        println!("nfa:\n{}\n", nfa3);
        let (d_tran, matches) = d_tran(&nfa3);
        println!("d_tran:\n{:?}\n", d_tran);
        println!("matches:\n{:?}\n", matches);
        println!("dfa:\n{}\n", nfa_to_dfa(&nfa3));
    }
}
