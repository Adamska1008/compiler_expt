use compiler_expt::charset::{range, CharSetTable};
use compiler_expt::graph::{closure, generate_basic_nfa, nfa_to_dfa, product, union, DriverType};
use std::rc::Rc;

// test of (a|b)*abb
fn test() {
    let mut table = CharSetTable::new();
    let sa_id = range('a', 'a', &mut table);
    let sb_id = range('b', 'b', &mut table);
    let table = Rc::new(table);
    let a = generate_basic_nfa(DriverType::Charset, sa_id, &table);
    let b = generate_basic_nfa(DriverType::Charset, sb_id, &table);
    let a_or_b = union(&a, &b);
    let a_or_b_closure = closure(&a_or_b);
    let a_or_b_closure_a = product(&a_or_b_closure, &a);
    let a_or_b_closure_a_b = product(&a_or_b_closure_a, &b);
    let a_or_b_closure_a_b_b = product(&a_or_b_closure_a_b, &b);
    println!("nfa:\n{}", a_or_b_closure_a_b_b);
    println!("dfa:\n{}", nfa_to_dfa(&a_or_b_closure_a_b_b));
}

fn main() {
    test()
}
