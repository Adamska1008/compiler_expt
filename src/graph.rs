use super::reg::CharSetTable;
use std::f32::consts::E;
use std::slice::Iter;

// NFA词素类型
#[derive(Copy, Clone, Debug)]
pub enum LexemeCategory {
    Nil,             // 不构成词素
    IntegerConst,    // 整数常量
    FloatConst,      // 实数常量
    ScientificConst, // 科学计数法常量
    NumericConst,    // 数值运算词
    Note,            // 注释
    StringConst,     // 字符串常量
    SpaceConst,      // 空格常量
    CompareOperator, // 比较运算词
    Ident,           // 变量
    LogicOperator,   // 逻辑运算词
}

#[derive(Copy, Clone, Debug)]
pub enum DriverType {
    Null,
    Char,
    Charset,
}

#[derive(Copy, Clone, Debug)]
pub enum StateType {
    Match,
    Unmatch,
}

// NFA 图
// 确保0为开始状态，最后一个状态为结束状态
// 有且仅有一个状态为结束状态
#[derive(Clone, Debug)]
pub struct Graph {
    graph_id: usize,
    edge_table: Vec<Edge>,
    state_table: Vec<State>,
    charset_table: CharSetTable,
}

impl Graph {
    pub fn new(
        edge_table: Vec<Edge>,
        state_table: Vec<State>,
        charset_table: CharSetTable,
    ) -> Self {
        Graph {
            graph_id: 0,
            edge_table,
            state_table,
            charset_table,
        }
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

    pub fn edges(&self) -> Iter<'_, Edge> {
        self.edge_table.iter()
    }

    pub fn states(&self) -> Iter<'_, State> {
        self.state_table.iter()
    }

    pub fn r#move(&self) {}
}

#[derive(Clone, Debug)]
pub struct Edge {
    from_state: usize,
    next_state: usize,
    driver_id: usize,
    driver_type: DriverType,
}

impl Edge {
    pub fn new(
        from_state: usize,
        next_state: usize,
        driver_id: usize,
        driver_type: DriverType,
    ) -> Self {
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

// 判断是否有出边
fn assert_out_edge(state_id: usize, graph: &Graph) -> bool {
    for edge in graph.edges() {
        if edge.from_state == state_id {
            return true;
        }
    }
    return false;
}

// 判断是否有入边
fn assert_in_edge(state_id: usize, graph: &Graph) -> bool {
    for edge in graph.edges() {
        if edge.next_state == state_id {
            return true;
        }
    }
    return false;
}

fn generate_basic_nfa(driver_type: DriverType, driver_id: usize, table: CharSetTable) -> Graph {
    Graph::new(
        vec![Edge::new(0, 1, driver_id, driver_type)],
        vec![State::new(0), State::new(1).set_type(StateType::Match)],
        table,
    )
}

// 并运算的构造法是直接修改状态编号
fn union(nfa1: &Graph, nfa2: &Graph) -> Graph {
    let size = nfa1.state_table.len();
    let mut new_graph = Graph::new(vec![], vec![], nfa1.charset_table.clone());
    new_graph.add_state(State::new(0));
    new_graph.add_state(State::new(size - 1).set_type(StateType::Match));
    // 对于nfa1，除了结束状态和开始状态，全部加到新图里
    // 对于边，将结束状态的编号修改为size-1
    new_graph.append_states(
        nfa1.states()
            .map(|s| s.clone())
            .filter(|state| state.state_id > 0 && state.state_id < nfa1.num_of_states() - 1)
            .collect(),
    );
    new_graph.append_edges(
        nfa1.edges()
            .map(|edge| {
                if edge.next_state == nfa1.num_of_states() - 1 {
                    Edge::new(edge.from_state, size - 1, edge.driver_id, edge.driver_type)
                } else {
                    edge.clone()
                }
            })
            .collect(),
    );
    // 对于nfa2，0状态和结束状态直接跳过；其他序号全部加nfa1.num_of_state-1
    new_graph.append_states(
        nfa2.states()
            .filter(|state| state.state_id >= 1 && state.state_id < nfa2.num_of_states() - 1)
            .map(|state| State::new(state.state_id + nfa1.num_of_states() - 1))
            .collect(),
    );
    new_graph.append_edges(
        nfa2.edges()
            .map(|edge| {
                let offset = nfa1.num_of_states() - 1;
                Edge::new(
                    if edge.from_state == 0 {
                        0
                    } else {
                        edge.from_state + offset
                    },
                    edge.next_state + offset,
                    edge.driver_id,
                    edge.driver_type,
                )
            })
            .collect(),
    );

    new_graph
}

// 连接运算
// 先nfa1，再nfa2
fn product(nfa1: &Graph, nfa2: &Graph) -> Graph {
    let mut new_nfa = nfa1.clone();
    new_nfa.end_state().state_type = StateType::Unmatch;
    if assert_out_edge(nfa1.num_of_states() - 1, nfa1) && assert_in_edge(0, nfa2) {
        new_nfa.add_state(State::new(new_nfa.num_of_states()));
        new_nfa.add_edge(Edge::new(
            new_nfa.num_of_states() - 2,
            new_nfa.num_of_states() - 1,
            0,
            DriverType::Null,
        ));
    }
    new_nfa.append_states(
        nfa2.states()
            .map(|state| State::new(nfa1.num_of_states() + state.state_id))
            .collect(),
    );
    new_nfa.end_state().state_type = StateType::Match;
    new_nfa.append_edges(
        nfa2.edges()
            .map(|edge| {
                Edge::new(
                    edge.from_state + nfa1.num_of_states(),
                    edge.next_state + nfa2.num_of_states(),
                    edge.driver_id,
                    edge.driver_type,
                )
            })
            .collect(),
    );
    new_nfa
}

// 正闭包运算
// 直接在结束状态添加一个到开始状态的边
fn plus_closure(nfa: &Graph) -> Graph {
    let mut new_nfa = nfa.clone();
    new_nfa.add_edge(Edge::new(0, nfa.num_of_states() - 1, 0, DriverType::Null));
    new_nfa
}

// 闭包运算
// 分多种情况，最多加四条边
fn closure(nfa: &Graph) -> Graph {
    let mut new_nfa = nfa.clone();
    new_nfa.edge_table.push(Edge::new(
        new_nfa.num_of_states() - 1,
        0,
        0,
        DriverType::Null,
    ));
    if assert_in_edge(0, &new_nfa) {
        new_nfa
            .state_table
            .iter_mut()
            .for_each(|s| s.state_id = s.state_id + 1);
        new_nfa.state_table.insert(0, State::new(0));
        new_nfa
            .edge_table
            .push(Edge::new(0, 1, 0, DriverType::Null));
    }
    if assert_out_edge(new_nfa.num_of_states() - 1, &new_nfa) {
        new_nfa.end_state().state_type = StateType::Unmatch;
        new_nfa
            .state_table
            .push(State::new(new_nfa.num_of_states()).set_type(StateType::Match));
        new_nfa.edge_table.push(Edge::new(
            new_nfa.num_of_states() - 2,
            new_nfa.num_of_states() - 1,
            0,
            DriverType::Null,
        ));
    }

    new_nfa
}
// 0 或者 1 个运算。
fn zero_or_one(nfa: &Graph) -> Graph {
    let mut new_nfa = nfa.clone();
    if assert_in_edge(0, &new_nfa) {
        new_nfa
            .state_table
            .iter_mut()
            .for_each(|s| s.state_id = s.state_id + 1);
        new_nfa.state_table.insert(0, State::new(0));
        new_nfa
            .edge_table
            .push(Edge::new(0, 1, 0, DriverType::Null));
    }
    if assert_out_edge(new_nfa.num_of_states() - 1, &new_nfa) {
        new_nfa.end_state().state_type = StateType::Unmatch;
        new_nfa
            .state_table
            .push(State::new(new_nfa.num_of_states()).set_type(StateType::Match));
        new_nfa.edge_table.push(Edge::new(
            new_nfa.num_of_states() - 2,
            new_nfa.num_of_states() - 1,
            0,
            DriverType::Null,
        ));
    }
    new_nfa.edge_table.push(Edge::new(
        0,
        new_nfa.num_of_states() - 1,
        0,
        DriverType::Null,
    ));
    new_nfa
}

#[cfg(test)]
mod test {
    use crate::graph::DriverType::Charset;
    use crate::graph::{generate_basic_nfa, DriverType, Graph, union};
    use crate::reg::{CharSet, CharSetTable};

    #[test]
    fn union_test() {
        let mut charset_table = CharSetTable::new();
        charset_table.push(CharSet::new(0, 0, 'a', 'a'));
        charset_table.push(CharSet::new(1, 0, 'b', 'b'));
        let mut graph_a = generate_basic_nfa(DriverType::Charset, 0, charset_table.clone());
        let mut graph_b = generate_basic_nfa(DriverType::Charset, 1, charset_table.clone());
        let new_nfa = union(&graph_a, &graph_b);
        println!("{:?}", new_nfa);
    }
}
