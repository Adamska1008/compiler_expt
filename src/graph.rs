use super::reg::CharSetTable;

// NFA词素类型
pub enum LexemeCategory {
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

enum DriverType {
    Null,
    Char,
    Charset,
}

enum StateType {
    Match,
    Unmatch,
}

// NFA 图
pub struct Graph {
    graph_id: usize,
    num_of_states: usize,
    edge_table: Vec<Edge>,
    state_table: Vec<State>,
}

struct Edge {
    from_state: usize,
    next_state: usize,
    driver_id: usize,
    driver_type: DriverType,
}

struct State {
    state_id: usize,
    state_type: StateType,
    category: LexemeCategory,
}

// 并运算
fn generate_basic_nfa(driver_type: DriverType, driver_id: usize, table: &CharSetTable) -> Graph {
    todo!()
}

// 连接运算
fn union(nfa1: &Graph, nfa2: &Graph) -> Graph {
    todo!()
}

//正闭包运算
fn product(nfa1: &Graph, nfa2: &Graph) -> Graph {
    todo!()
}

// 闭包运算
fn plus_closure(nfa: &Graph) -> Graph {
    todo!()
}

// 0 或者 1 个运算。
fn closure(nfa: &Graph) -> Graph {
    todo!()
}

fn zero_or_one(nfa: &Graph) -> Graph {
    todo!()
}
