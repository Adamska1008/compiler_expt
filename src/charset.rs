use std::collections::HashSet;

// one segment of a charset, express only chars in a certain range
// to express a complete charset, use charset table instead
#[derive(Debug, Copy, Clone)]
pub struct CharSet {
    index_id: usize,   // identifier of charset
    segment_id: usize, // identifier of segment. one charset contains multiple segments
    from_char: char,
    to_char: char,
}

impl CharSet {
    pub fn index_id(&self) -> usize {
        self.index_id
    }

    pub fn new(index_id: usize, segment_id: usize, from_char: char, to_char: char) -> Self {
        Self {
            index_id,
            segment_id,
            from_char,
            to_char,
        }
    }
}

pub type CharSetTable = Vec<CharSet>;

// return all charset id
pub fn charsets_in(table: &CharSetTable) -> HashSet<usize> {
    table.iter().map(|seg| seg.index_id).collect()
}

// next identifier which is unused and available
fn peek_id(table: &CharSetTable) -> usize {
    match table.last() {
        Some(last) => last.index_id + 1,
        None => 0,
    }
}

// judge if ch is in the charset with set_id
pub fn contains(set_id: usize, ch: char, table: &CharSetTable) -> bool {
    for seg in table.iter().filter(|seg| seg.index_id == set_id) {
        if seg.from_char <= ch && seg.to_char >= ch {
            return true;
        }
    }
    return false;
}

// range operation: [<from>-<to>]
// when calling this function, should make sure from <= to
pub fn range(from_char: char, to_char: char, table: &mut CharSetTable) -> usize {
    let set_id = peek_id(table);
    table.push(CharSet::new(set_id, 0, from_char, to_char));
    set_id
}

// union operation: [<c1><c2>]
// when calling this function, should make sure c1 != c2
pub fn union_ch(c1: char, c2: char, table: &mut CharSetTable) -> usize {
    let set_id = peek_id(table);
    if c1 > c2 {
        // if c1 > c2, swap them
        return union_ch(c2, c1, table);
    }
    if c2 as u8 - c2 as u8 == 1 {
        // is c1 and c2 is consistent, switch to range
        return range(c1, c2, table);
    }
    // use two segments to express union relationship
    table.push(CharSet::new(set_id, 0, c1, c1));
    table.push(CharSet::new(set_id, 1, c2, c2));
    return set_id;
}

// union operation: [<set><ch>]
pub fn union_mix(set_id: usize, ch: char, table: &mut CharSetTable) -> usize {
    let next_id = peek_id(table);
    // collect charset from old set_id
    let mut new_set: Vec<_> = table
        .iter()
        .filter(|seg| seg.index_id == set_id)
        .map(|seg| CharSet::new(next_id, seg.segment_id, seg.from_char, seg.to_char))
        .collect();
    // If the old set do not contains the new character(likely), add it
    if !contains(set_id, ch, table) {
        let seg_id = new_set.iter().max_by_key(|seg| seg.segment_id).unwrap().segment_id + 1;
        new_set.push(CharSet::new(next_id, seg_id, ch, ch));
    }
    table.append(&mut new_set);
    next_id
}

// union operation: [<set><set>]
pub fn union_set(set_id_l: usize, set_id_r: usize, table: &mut CharSetTable) -> usize {
    let next_id = peek_id(table);
    let mut new_set: Vec<_> = table
        .iter()
        .filter(|seg| seg.index_id == set_id_l || seg.index_id == set_id_r)
        .map(|seg| CharSet::new(next_id, seg.segment_id, seg.from_char, seg.to_char))
        .collect();
    table.append(&mut new_set);
    next_id
}

// difference operation
pub fn diff_set_ch(set_id: usize, ch: char, table: &mut CharSetTable) -> usize {
    let next_id = peek_id(table);
    let mut seg_id = 0;
    let mut new_set = vec![];
    for seg in table.iter().filter(|s| s.index_id == set_id) {
        // if ch == one_char_segment, just delete(ignore) the segment
        if ch == seg.from_char && ch != seg.to_char {
            // if ch is the left side of segment, shrink the segment
            new_set.push(CharSet::new(next_id, seg_id, (ch as u8 + 1) as char, seg.to_char));
            seg_id += 1;
        } else if ch == seg.to_char && ch != seg.from_char {
            // same when ch is the right side of segment
            new_set.push(CharSet::new(next_id, seg_id, seg.from_char, (ch as u8 - 1) as char));
            seg_id += 1;
        } else if ch > seg.from_char && ch < seg.to_char {
            // if ch is in the segment, spilt the segment into two
            new_set.append(&mut vec![
                CharSet::new(next_id, seg_id, (ch as u8 + 1) as char, seg.to_char),
                CharSet::new(next_id, seg_id + 1, seg.from_char, (ch as u8 - 1) as char),
            ]);
            seg_id += 2;
        }
    }
    table.append(&mut new_set);
    next_id
}

#[cfg(test)]
mod test {
    use crate::charset::{range, union_ch, union_mix, union_set, CharSetTable};

    #[test]
    fn operation_test() {
        let mut table = CharSetTable::new();
        let sta_id = range('a', 'c', &mut table);
        let stb_id = union_ch('f', 'k', &mut table);
        let stc_id = union_set(sta_id, stb_id, &mut table);
        let _ = union_mix(stc_id, 'q', &mut table);
        println!("{:?}", table);
    }
}
