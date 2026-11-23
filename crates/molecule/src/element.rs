//! Chemical elements with radii and display colours.

/// A chemical element, identified by its symbol in a static table.
/// `Element::UNKNOWN` is used for anything the table does not cover.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Element(u8);

struct Info {
    symbol: &'static str,
    /// Single-bond covalent radius, Å (Cordero et al. 2008).
    covalent: f32,
    /// Van der Waals radius, Å (Bondi / Alvarez).
    vdw: f32,
    /// Jmol-style CPK colour, sRGB.
    color: [u8; 3],
}

const fn info(symbol: &'static str, covalent: f32, vdw: f32, color: u32) -> Info {
    Info {
        symbol,
        covalent,
        vdw,
        color: [(color >> 16) as u8, (color >> 8) as u8, color as u8],
    }
}

// Index 0 is the "unknown" element.
static TABLE: &[Info] = &[
    info("X", 0.77, 1.70, 0xFF1493),
    info("H", 0.31, 1.10, 0xFFFFFF),
    info("He", 0.28, 1.40, 0xD9FFFF),
    info("Li", 1.28, 1.82, 0xCC80FF),
    info("Be", 0.96, 1.53, 0xC2FF00),
    info("B", 0.84, 1.92, 0xFFB5B5),
    info("C", 0.76, 1.70, 0x909090),
    info("N", 0.71, 1.55, 0x3050F8),
    info("O", 0.66, 1.52, 0xFF0D0D),
    info("F", 0.57, 1.47, 0x90E050),
    info("Ne", 0.58, 1.54, 0xB3E3F5),
    info("Na", 1.66, 2.27, 0xAB5CF2),
    info("Mg", 1.41, 1.73, 0x8AFF00),
    info("Al", 1.21, 1.84, 0xBFA6A6),
    info("Si", 1.11, 2.10, 0xF0C8A0),
    info("P", 1.07, 1.80, 0xFF8000),
    info("S", 1.05, 1.80, 0xFFFF30),
    info("Cl", 1.02, 1.75, 0x1FF01F),
    info("Ar", 1.06, 1.88, 0x80D1E3),
    info("K", 2.03, 2.75, 0x8F40D4),
    info("Ca", 1.76, 2.31, 0x3DFF00),
    info("Mn", 1.39, 2.05, 0x9C7AC7),
    info("Fe", 1.32, 2.04, 0xE06633),
    info("Co", 1.26, 2.00, 0xF090A0),
    info("Ni", 1.24, 1.97, 0x50D050),
    info("Cu", 1.32, 1.96, 0xC88033),
    info("Zn", 1.22, 2.01, 0x7D80B0),
    info("Se", 1.20, 1.90, 0xFFA100),
    info("Br", 1.20, 1.85, 0xA62929),
    info("Cd", 1.44, 2.18, 0xFFD98F),
    info("I", 1.39, 1.98, 0x940094),
    info("Pt", 1.36, 2.13, 0xD0D0E0),
    info("Au", 1.36, 2.14, 0xFFD123),
    info("Hg", 1.32, 2.23, 0xB8B8D0),
];

impl Element {
    pub const UNKNOWN: Element = Element(0);
    pub const H: Element = Element(1);
    pub const C: Element = Element(6);

    /// Case-insensitive lookup (`"CL"`, `"Cl"` and `"cl"` all work).
    pub fn from_symbol(symbol: &str) -> Element {
        let s = symbol.trim();
        TABLE
            .iter()
            .position(|e| e.symbol.eq_ignore_ascii_case(s))
            .map(|i| Element(i as u8))
            .unwrap_or(Element::UNKNOWN)
    }

    /// Guess the element from a PDB atom name (columns 13-16) when the element
    /// column is missing: `" CA "` is C-alpha (carbon), `"CA  "` is calcium.
    pub fn guess_from_atom_name(name: &str) -> Element {
        let trimmed = name.trim();
        if name.len() >= 2 && !name.starts_with(' ') && !name.starts_with(|c: char| c.is_ascii_digit()) {
            let two = Element::from_symbol(&trimmed[..trimmed.len().min(2)]);
            if two != Element::UNKNOWN && trimmed.len() >= 2 {
                return two;
            }
        }
        let first = trimmed.trim_start_matches(|c: char| c.is_ascii_digit());
        first
            .get(..1)
            .map(Element::from_symbol)
            .unwrap_or(Element::UNKNOWN)
    }

    fn info(self) -> &'static Info {
        &TABLE[self.0 as usize]
    }

    pub fn symbol(self) -> &'static str {
        self.info().symbol
    }

    pub fn covalent_radius(self) -> f32 {
        self.info().covalent
    }

    pub fn vdw_radius(self) -> f32 {
        self.info().vdw
    }

    /// sRGB colour in `0.0..=1.0`.
    pub fn color(self) -> [f32; 3] {
        self.info().color.map(|c| c as f32 / 255.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_is_case_insensitive() {
        assert_eq!(Element::from_symbol("CL").symbol(), "Cl");
        assert_eq!(Element::from_symbol(" n ").symbol(), "N");
        assert_eq!(Element::from_symbol("Qq"), Element::UNKNOWN);
    }

    #[test]
    fn guesses_from_pdb_atom_names() {
        assert_eq!(Element::guess_from_atom_name(" CA ").symbol(), "C");
        assert_eq!(Element::guess_from_atom_name("CA  ").symbol(), "Ca");
        assert_eq!(Element::guess_from_atom_name("1HB "), Element::H);
    }
}
