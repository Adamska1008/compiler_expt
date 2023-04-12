use crate::charset::CharSetTable;

use std::cell::RefCell;

use super::graph::{Graph, LexemeCategory};

// NFA 操作数类型
enum OperandType {
    Char,
    Charset,
    Regular,
    Nil, // used when there's no operand
}

// regular infix or prefix Expression
// unable to express a complete expression, use table instead
pub struct RegularExpression {
    regular_id: usize,
    name: String,
    operator_symbol: char,    // ‘=’, ‘~’, ‘-’, ‘|’, ‘.’, ‘*’, ‘+’, ‘?’
    operand_id_l: usize,      // identifier of left operand, store in one certain charset table
    operand_id_r: usize,      //
    type_l: OperandType,      // type of left operand, possible: Char, Charset, Regular, Nil ( Nil for no operand)
    type_r: OperandType,      //
    result_type: OperandType, // type of operation result
    category: LexemeCategory, // type of lexeme
}

pub struct RegularTable {
    inner: Vec<RegularExpression>,
    charset: RefCell<CharSetTable>,
    dfa: Graph,
}

impl RegularTable {
    // store the vector and build corresponding dfa
    pub fn from(inner: Vec<RegularExpression>) -> Self {
        todo!()
    }

    // test if the str matches pattern dfa
    pub fn matches(&self, s: &str) -> bool {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use crate::charset::{range, union_ch, union_mix, union_set, CharSetTable};
    use crate::graph::LexemeCategory;
    use crate::reg::{OperandType, RegularExpression, RegularTable};

    #[test]
    fn transform_test() {
        let example = vec![
            RegularExpression {
                regular_id: 1,
                name: "r1".to_string(),
                operator_symbol: '|',
                operand_id_l: 'a' as usize,
                operand_id_r: 'b' as usize,
                type_l: OperandType::Char,
                type_r: OperandType::Char,
                result_type: OperandType::Charset,
                category: LexemeCategory::Nil,
            },
            RegularExpression {
                regular_id: 2,
                name: "r2".to_string(),
                operator_symbol: '*',
                operand_id_l: 1,
                operand_id_r: 0,
                type_l: OperandType::Charset,
                type_r: OperandType::Nil,
                result_type: OperandType::Regular,
                category: LexemeCategory::Nil,
            },
            RegularExpression {
                regular_id: 3,
                name: "r3".to_string(),
                operator_symbol: '.',
                operand_id_l: 2,
                operand_id_r: 'a' as usize,
                type_l: OperandType::Regular,
                type_r: OperandType::Char,
                result_type: OperandType::Regular,
                category: LexemeCategory::Nil,
            },
            RegularExpression {
                regular_id: 4,
                name: "r4".to_string(),
                operator_symbol: '.',
                operand_id_l: 3,
                operand_id_r: 'b' as usize,
                type_l: OperandType::Regular,
                type_r: OperandType::Char,
                result_type: OperandType::Regular,
                category: LexemeCategory::Nil,
            },
            RegularExpression {
                regular_id: 4,
                name: "r4".to_string(),
                operator_symbol: '.',
                operand_id_l: 3,
                operand_id_r: 'b' as usize,
                type_l: OperandType::Regular,
                type_r: OperandType::Char,
                result_type: OperandType::Regular,
                category: LexemeCategory::Nil,
            },
        ];
        let regular_table = RegularTable::from(example);
        println!("{}", regular_table.matches("abababb"));
    }
}
