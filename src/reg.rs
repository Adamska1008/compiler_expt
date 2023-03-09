use super::graph::{Graph, LexemeCategory};

// NFA 操作数类型
enum OperandType {
    CHAR,    // 字符
    CHARSET, // 字符集
    REGULAR, // 正则表达式
}

// 正则表达式
struct RegularExpression {
    regular_id: usize,
    name: String,
    operator_symbol: char, //正则运算符，共有 7 种：‘=’, ‘~’, ‘-’, ‘|’, ‘.’, ‘*’, ‘+’, ‘?’
    operator_id_l: usize,  //左操作数
    operator_id_r: usize,  //右操作数
    type_l: OperandType,   //左操作数的类型
    type_r: OperandType,   //右操作数的类型
    result_type: OperandType, //运算结果的类型
    category: LexemeCategory, // 词的 category 属性值
    nfa: Graph,            //对应的 NFA
}

// 字符集
struct CharSet {
    index_id: usize,   //字符集 id
    segment_id: usize, //字符集中的段 id。一个字符集可以包含多个段
    from_char: char,   //段的起始字符
    to_char: char,     //段的结尾字符
}

impl CharSet {
    pub fn new(index_id: usize, segment_id: usize, from_char: char, to_char: char) -> Self {
        Self {
            index_id,
            segment_id,
            from_char,
            to_char,
        }
    }
}

pub type RegularTable = Vec<RegularExpression>;
pub type CharSetTable = Vec<CharSet>;

// 字符集表的下一个字符集id
fn peek_id(table: &CharSetTable) -> usize {
    match table.last() {
        Some(last) => last.index_id + 1,
        None => 0,
    }
}

// 字符的范围运算
// 假定确保 from_char <= to_char
fn range(from_char: char, to_char: char, mut table: &mut CharSetTable) -> usize {
    let set_id = peek_id(table);
    table.push(CharSet::new(set_id, 0, from_char, to_char));
    set_id
}

// 字符的并运算
// 假定确保 c1 != c2
// 当两字符相连时，等价于range(c1, c2)
// 否则创建包含两段的字符集，每个字符集包含一个字符
fn union_ch(c1: char, c2: char, mut table: &mut CharSetTable) -> usize {
    let set_id = peek_id(table);
    if c1 > c2 {
        return union_ch(c2, c1, table);
    }
    if c2 as u8 - c2 as u8 == 1 {
        return range(c1, c2, table);
    }
    table.push(CharSet::new(set_id, 0, c1, c1));
    table.push(CharSet::new(set_id, 1, c2, c2));
    return set_id;
}

// 字符和字符集的并运算
// 创建新的字符集，原字符集不变
// 不检测给定字符是否在原字符集范围之内
fn union_mix(set_id: usize, c: char, mut table: &mut CharSetTable) -> usize {
    let next_id = peek_id(table);
    let mut new_set: Vec<_> = table
        .iter()
        .filter(|seg| seg.index_id == set_id)
        .map(|mut seg| CharSet::new(next_id, seg.segment_id, seg.from_char, seg.to_char))
        .collect();
    let seg_id = new_set
        .iter()
        .max_by_key(|seg| seg.segment_id)
        .unwrap()
        .segment_id
        + 1;
    new_set.push(CharSet::new(next_id, seg_id, c, c));
    table.append(&mut new_set);
    next_id
}

// 字符集的并运算
// 创建新的字符集，原字符集不变
// 不检测两字符集是否有交叉
fn union_set(set_id_l: usize, set_id_r: usize, mut table: &mut CharSetTable) -> usize {
    let next_id = peek_id(table);
    let mut new_set: Vec<_> = table
        .iter()
        .filter(|seg| seg.index_id == set_id_l || seg.index_id == set_id_r)
        .map(|mut seg| CharSet::new(next_id, seg.segment_id, seg.from_char, seg.to_char))
        .collect();
    table.append(&mut new_set);
    next_id
}

// 字符集的减运算
// 创建新的字符集，原字符集不变
// 基本思路：
// 对每个段分别检测并处理
// 有如下可能性
// 1. 这个段刚好等于这个字符，直接删除段
// 2. 这个段减去这个字符后拆分为两个段
// 3. 这个字符刚好是段的from或to, 最后仍是一个段
fn diff_set_ch(set_id: usize, c: char, mut table: &mut CharSetTable) -> usize {
    let next_id = peek_id(table);
    let mut seg_id = 0;
    let mut new_set = vec![];
    for seg in table.iter() {
        if c == seg.from_char {
            new_set.push(CharSet::new(
                next_id,
                seg_id,
                (c as u8 + 1) as char,
                seg.to_char,
            ));
            seg_id += 1;
        } else if c == seg.to_char {
            new_set.push(CharSet::new(
                next_id,
                seg_id,
                seg.from_char,
                (c as u8 - 1) as char,
            ));
            seg_id += 1;
        } else if c > seg.from_char && c < seg.to_char {
            new_set.append(&mut vec![
                CharSet::new(next_id, seg_id, (c as u8 + 1) as char, seg.to_char),
                CharSet::new(next_id, seg_id + 1, seg.from_char, (c as u8 - 1) as char),
            ]);
            seg_id += 2;
        }
    }
    table.append(&mut new_set);
    next_id
}
