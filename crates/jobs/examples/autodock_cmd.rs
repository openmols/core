//! Print the AutoDock job script for given settings (to run by hand in the image).
//!
//! `cargo run -p jobs --example autodock_cmd -- vina|ad4 cx cy cz [A:CYS:481 acrylamide]`

use jobs::autodock::{Covalent, Engine, Settings, Warhead, command};

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let engine = if a[0] == "ad4" { Engine::AutoDock4 } else { Engine::Vina };
    let num = |i: usize| a[i].parse::<f32>().unwrap();
    let covalent = a.get(4).map(|residue| Covalent {
        residue: residue.clone(),
        warhead: a.get(5).and_then(|w| Warhead::ALL.into_iter().find(|x| x.key() == w)).unwrap_or_default(),
    });
    let settings = Settings {
        engine,
        center: [num(1), num(2), num(3)],
        size: [22.0; 3],
        num_modes: if engine == Engine::AutoDock4 { 5 } else { 5 },
        ad4_evals: 1_000_000,
        covalent,
        ..Default::default()
    };
    println!("{}", command(&settings).unwrap());
}
