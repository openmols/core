//! Parse a finished AutoDock job folder and print its poses.
//! `cargo run -p jobs --example autodock_poses -- <job dir> [no-scores]`
use jobs::autodock::{AD4_OUT, VINA_OUT, parse_dlg, parse_vina};

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let scores = std::env::args().nth(2).as_deref() != Some("no-scores");
    let poses = if dir.join(AD4_OUT).is_file() {
        parse_dlg(&std::fs::read_to_string(dir.join(AD4_OUT)).unwrap(), !scores).unwrap()
    } else {
        parse_vina(&std::fs::read_to_string(dir.join(VINA_OUT)).unwrap(), scores).unwrap()
    };
    for p in &poses.poses {
        println!("rank {} energy {:?} at {:.1?}", p.rank, p.energy, p.position);
    }
    std::fs::write(dir.join("poses.pdb"), &poses.pdb).unwrap();
}
