//! Print the FRODOCK pipeline openmols would run, for testing it by hand.
//!
//! cargo run -p jobs --example frodock_cmd -- [antibody|enzyme|other] [top_n]

use jobs::frodock::{Interaction, Settings, command};

fn main() {
    let mut args = std::env::args().skip(1);
    let interaction = match args.next().as_deref() {
        Some("antibody") => Interaction::Antibody,
        Some("enzyme") => Interaction::Enzyme,
        _ => Interaction::Other,
    };
    let top_n = args.next().and_then(|v| v.parse().ok()).unwrap_or(10);
    print!("{}", command(&Settings { interaction, top_n, ..Default::default() }));
}
