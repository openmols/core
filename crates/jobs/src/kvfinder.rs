//! pyKVFinder job: command line and results parsing.
//!
//! Inputs (written by the app into the job folder): `receptor.pdb` (polymer
//! only — hetero groups would fill the cavities) and optionally `ligand.pdb`
//! for ligand-adjusted detection. Outputs land in `out/`:
//! `receptor.KVFinder.results.toml` and `receptor.KVFinder.output.pdb`
//! (cavity grid points; residue name = cavity id, B-factor = depth).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

pub const RESULTS_FILE: &str = "out/receptor.KVFinder.results.toml";
pub const CAVITY_PDB: &str = "out/receptor.KVFinder.output.pdb";

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Settings {
    /// Probe-out radius, Å (bigger finds shallower pockets).
    pub probe_out: f32,
    /// Removal distance, Å.
    pub removal_distance: f32,
    /// Minimum cavity volume, Å³.
    pub volume_cutoff: f32,
    /// Restrict to the region around `ligand.pdb`.
    pub ligand_adjust: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            probe_out: 4.0,
            removal_distance: 2.4,
            volume_cutoff: 5.0,
            ligand_adjust: false,
        }
    }
}

pub fn command(settings: &Settings) -> String {
    let mut cmd = format!(
        "pyKVFinder receptor.pdb -D --hydropathy -O out -o {} -R {} -V {}",
        settings.probe_out, settings.removal_distance, settings.volume_cutoff
    );
    if settings.ligand_adjust {
        cmd.push_str(" -L ligand.pdb");
    }
    cmd
}

/// Where a frame's results land inside the job folder.
pub fn frame_results(frame: usize) -> String {
    format!("out/{}/frame.KVFinder.results.toml", frame_name(frame))
}

pub fn frame_name(frame: usize) -> String {
    format!("f{frame:05}")
}

/// Run pyKVFinder over every frame written to `frames/`, one container, one
/// output folder per frame. `--base_name frame` keeps the output names
/// identical whatever the input file is called.
pub fn ensemble_command(settings: &Settings) -> String {
    format!(
        "set -e\n\
         for f in frames/*.pdb; do\n\
         \x20 n=$(basename \"$f\" .pdb)\n\
         \x20 pyKVFinder \"$f\" -b frame -D --hydropathy -O \"out/$n\" -o {} -R {} -V {} >> kvfinder.log 2>&1\n\
         \x20 echo \"done $n\"\n\
         done",
        settings.probe_out, settings.removal_distance, settings.volume_cutoff
    )
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LiningResidue {
    pub number: String,
    pub chain: String,
    pub name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Cavity {
    /// `KAA`, `KAB`, … (also the residue name in the cavity PDB).
    pub id: String,
    pub volume: f64,
    pub area: f64,
    pub max_depth: Option<f64>,
    pub avg_depth: Option<f64>,
    pub avg_hydropathy: Option<f64>,
    pub residues: Vec<LiningResidue>,
}

impl Cavity {
    pub fn residue_labels(&self) -> String {
        self.residues
            .iter()
            .map(|r| format!("{}:{} {}", r.chain, r.name, r.number))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Deserialize)]
struct ResultsFile {
    #[serde(rename = "RESULTS")]
    results: Results,
}

#[derive(Deserialize)]
struct Results {
    #[serde(rename = "VOLUME", default)]
    volume: BTreeMap<String, f64>,
    #[serde(rename = "AREA", default)]
    area: BTreeMap<String, f64>,
    #[serde(rename = "MAX_DEPTH", default)]
    max_depth: BTreeMap<String, f64>,
    #[serde(rename = "AVG_DEPTH", default)]
    avg_depth: BTreeMap<String, f64>,
    /// Also contains the scale range (an array), so parse values loosely.
    #[serde(rename = "AVG_HYDROPATHY", default)]
    avg_hydropathy: BTreeMap<String, toml::Value>,
    #[serde(rename = "RESIDUES", default)]
    residues: BTreeMap<String, Vec<Vec<String>>>,
}

/// Parse `results.toml`; cavities sorted by volume, largest first.
pub fn parse_results(text: &str) -> Result<Vec<Cavity>> {
    let file: ResultsFile = toml::from_str(text).context("parsing pyKVFinder results")?;
    let r = file.results;
    let mut cavities: Vec<Cavity> = r
        .volume
        .iter()
        .map(|(id, volume)| Cavity {
            id: id.clone(),
            volume: *volume,
            area: r.area.get(id).copied().unwrap_or(0.0),
            max_depth: r.max_depth.get(id).copied(),
            avg_depth: r.avg_depth.get(id).copied(),
            avg_hydropathy: r.avg_hydropathy.get(id).and_then(|v| v.as_float()),
            residues: r
                .residues
                .get(id)
                .map(|list| {
                    list.iter()
                        .filter(|t| t.len() >= 3)
                        .map(|t| LiningResidue {
                            number: t[0].clone(),
                            chain: t[1].clone(),
                            name: t[2].clone(),
                        })
                        .collect()
                })
                .unwrap_or_default(),
        })
        .collect();
    cavities.sort_by(|a, b| b.volume.total_cmp(&a.volume));
    Ok(cavities)
}

pub fn read_results(job_dir: &Path) -> Result<Vec<Cavity>> {
    let path = job_dir.join(RESULTS_FILE);
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    parse_results(&text)
}

pub fn cavity_pdb(job_dir: &Path) -> PathBuf {
    job_dir.join(CAVITY_PDB)
}

/// The cavity PDB normalised for strict readers.
///
/// pyKVFinder writes per-point hydropathy into the occupancy column (can be
/// negative) and stops at column 66, so lines lack the element column.
/// Occupancy becomes 1.00, depth stays in the B-factor column and the element
/// (H) is added in columns 77-78.
pub fn read_cavity_pdb(job_dir: &Path) -> Result<String> {
    let path = cavity_pdb(job_dir);
    let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    Ok(sanitize_cavity_pdb(&text))
}

pub fn sanitize_cavity_pdb(text: &str) -> String {
    let mut out = String::with_capacity(text.len() * 5 / 4);
    // pyKVFinder gives every cavity the same residue number; renumber so each
    // cavity (KAA, KAB, …) is its own residue.
    let mut ids: Vec<String> = Vec::new();
    for line in text.lines() {
        if (line.starts_with("ATOM") || line.starts_with("HETATM")) && line.len() >= 54 {
            let id = line[17..20].to_string();
            let number = match ids.iter().position(|x| *x == id) {
                Some(i) => i + 1,
                None => {
                    ids.push(id);
                    ids.len()
                }
            };
            let bfactor = line.get(60..66).unwrap_or("  0.00");
            out.push_str(&format!(
                "{}{:>4}{}  1.00{:>6}          {:>2}\n",
                &line[..22],
                number % 10_000,
                &line[26..54],
                bfactor.trim(),
                "H"
            ));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

// ------------------------------------------------------- ensemble analysis

/// Per-residue values from analysis, for filters that go beyond the cavity's
/// own geometry: property name → `(chain, residue number)` → value.
///
/// The app fills this with whatever it has computed — RMSF and B-factor
/// today, anything an MD analysis produces later — and a filter refers to a
/// property by name, so new measures need no change here.
pub type ResidueProperties = BTreeMap<String, HashMap<(String, String), f64>>;

/// Something measurable about a cavity, for filtering and ranking.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Metric {
    Volume,
    Area,
    MaxDepth,
    AvgDepth,
    Hydropathy,
    ResidueCount,
    /// Mean over the lining residues of a named per-residue property, e.g.
    /// `rmsf` or `bfactor`. Unknown names make the cavity fail the filter.
    ResidueMean(String),
    /// Fraction of frames a merged pocket appears in (0..=1). Only meaningful
    /// after merging, so it is ignored when filtering individual cavities.
    Persistence,
}

impl Metric {
    /// Parse a filter name: the built-ins, or `mean:<property>` / any other
    /// name, which is looked up in [`ResidueProperties`].
    pub fn parse(name: &str) -> Metric {
        match name.trim().to_lowercase().as_str() {
            "volume" => Metric::Volume,
            "area" => Metric::Area,
            "max_depth" | "depth" => Metric::MaxDepth,
            "avg_depth" => Metric::AvgDepth,
            "hydropathy" | "avg_hydropathy" => Metric::Hydropathy,
            "residues" | "residue_count" => Metric::ResidueCount,
            "persistence" => Metric::Persistence,
            other => Metric::ResidueMean(other.trim_start_matches("mean:").to_string()),
        }
    }

    /// Every metric a cavity carries on its own, for listing in the panel.
    pub fn builtin() -> Vec<Metric> {
        vec![
            Metric::Volume,
            Metric::Area,
            Metric::MaxDepth,
            Metric::AvgDepth,
            Metric::Hydropathy,
            Metric::ResidueCount,
        ]
    }

    /// The name [`Metric::parse`] accepts, so filters round-trip as text.
    pub fn filter_name(&self) -> String {
        match self {
            Metric::Volume => "volume".into(),
            Metric::Area => "area".into(),
            Metric::MaxDepth => "max_depth".into(),
            Metric::AvgDepth => "avg_depth".into(),
            Metric::Hydropathy => "hydropathy".into(),
            Metric::ResidueCount => "residues".into(),
            Metric::Persistence => "persistence".into(),
            Metric::ResidueMean(p) => p.clone(),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Metric::Volume => "volume".into(),
            Metric::Area => "area".into(),
            Metric::MaxDepth => "max depth".into(),
            Metric::AvgDepth => "avg depth".into(),
            Metric::Hydropathy => "hydropathy".into(),
            Metric::ResidueCount => "residues".into(),
            Metric::Persistence => "persistence".into(),
            Metric::ResidueMean(p) => format!("mean {p}"),
        }
    }

    /// The cavity's value, or `None` when it cannot be determined.
    pub fn of(&self, c: &Cavity, properties: &ResidueProperties) -> Option<f64> {
        match self {
            Metric::Volume => Some(c.volume),
            Metric::Area => Some(c.area),
            Metric::MaxDepth => c.max_depth,
            Metric::AvgDepth => c.avg_depth,
            Metric::Hydropathy => c.avg_hydropathy,
            Metric::ResidueCount => Some(c.residues.len() as f64),
            Metric::Persistence => None,
            Metric::ResidueMean(property) => {
                let values = properties.get(property)?;
                let found: Vec<f64> = c
                    .residues
                    .iter()
                    .filter_map(|r| values.get(&(r.chain.clone(), r.number.clone())).copied())
                    .collect();
                (!found.is_empty()).then(|| found.iter().sum::<f64>() / found.len() as f64)
            }
        }
    }
}

/// One `min <= metric <= max` rule. Either bound may be left open.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Criterion {
    pub metric: Metric,
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl Criterion {
    pub fn at_least(metric: Metric, min: f64) -> Self {
        Self { metric, min: Some(min), max: None }
    }
    pub fn at_most(metric: Metric, max: f64) -> Self {
        Self { metric, min: None, max: Some(max) }
    }
    pub fn between(metric: Metric, min: f64, max: f64) -> Self {
        Self { metric, min: Some(min), max: Some(max) }
    }

    pub fn accepts(&self, value: f64) -> bool {
        self.min.is_none_or(|m| value >= m) && self.max.is_none_or(|m| value <= m)
    }

    pub fn describe(&self) -> String {
        match (self.min, self.max) {
            (Some(a), Some(b)) => format!("{} {a}–{b}", self.metric.label()),
            (Some(a), None) => format!("{} ≥ {a}", self.metric.label()),
            (None, Some(b)) => format!("{} ≤ {b}", self.metric.label()),
            (None, None) => self.metric.label(),
        }
    }
}

/// Which cavities survive, and when two of them count as the same pocket.
///
/// The defaults reproduce Karatzas, Brotzakis & Sarimveis (JCTC 2025) for
/// AR-V7: more than 5 lining residues, hydrophilic (hydropathy < 0),
/// area 80–2480 Å², volume above 120 Å³, merged at 60% shared residues.
/// Add criteria for anything else, including MD-derived per-residue
/// properties such as `rmsf`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnsembleFilter {
    pub criteria: Vec<Criterion>,
    /// Fraction of lining residues two cavities must share to be the same
    /// pocket (0.6 in that paper).
    pub residue_overlap: f64,
}

impl Default for EnsembleFilter {
    fn default() -> Self {
        Self {
            criteria: vec![
                Criterion::at_least(Metric::ResidueCount, 6.0),
                Criterion::at_most(Metric::Hydropathy, 0.0),
                Criterion::between(Metric::Area, 80.0, 2480.0),
                Criterion::at_least(Metric::Volume, 120.0),
            ],
            residue_overlap: 0.6,
        }
    }
}

impl EnsembleFilter {
    /// No filtering at all; useful as a starting point for custom rules.
    pub fn unfiltered() -> Self {
        Self { criteria: Vec::new(), residue_overlap: 0.6 }
    }

    pub fn with(mut self, criterion: Criterion) -> Self {
        self.criteria.retain(|c| c.metric != criterion.metric);
        self.criteria.push(criterion);
        self
    }

    pub fn without(mut self, metric: &Metric) -> Self {
        self.criteria.retain(|c| &c.metric != metric);
        self
    }

    /// Does this cavity pass every criterion? A criterion whose value cannot
    /// be determined (e.g. hydropathy was not computed, or an unknown
    /// property) rejects the cavity, so filters never pass silently.
    pub fn keeps_with(&self, c: &Cavity, properties: &ResidueProperties) -> bool {
        self.criteria
            .iter()
            .filter(|crit| crit.metric != Metric::Persistence)
            .all(|crit| crit.metric.of(c, properties).is_some_and(|v| crit.accepts(v)))
    }

    pub fn keeps(&self, c: &Cavity) -> bool {
        self.keeps_with(c, &ResidueProperties::new())
    }

    /// Post-merge criteria (currently just persistence).
    fn keeps_pocket(&self, pocket: &ConsensusPocket) -> bool {
        self.criteria
            .iter()
            .filter(|crit| crit.metric == Metric::Persistence)
            .all(|crit| crit.accepts(pocket.persistence))
    }

    pub fn describe(&self) -> String {
        if self.criteria.is_empty() {
            return "no filters".into();
        }
        self.criteria
            .iter()
            .map(Criterion::describe)
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Per-residue properties this filter needs that were not supplied.
    ///
    /// Any name that is not a built-in metric is read as a property, so a
    /// typo becomes a filter nothing can satisfy. Reporting the names makes
    /// that a correctable message rather than an empty result.
    pub fn unknown_properties(&self, properties: &ResidueProperties) -> Vec<String> {
        let mut missing: Vec<String> = self
            .criteria
            .iter()
            .filter_map(|c| match &c.metric {
                Metric::ResidueMean(p) if !properties.contains_key(p) => Some(p.clone()),
                _ => None,
            })
            .collect();
        missing.sort();
        missing.dedup();
        missing
    }

    /// The filter as editable text, one rule per line. Round-trips through
    /// [`EnsembleFilter::parse_text`].
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        for c in &self.criteria {
            let name = c.metric.filter_name();
            let line = match (c.min, c.max) {
                (Some(a), Some(b)) if a == b => format!("{name} = {a}"),
                (Some(a), Some(b)) => format!("{name} {a} .. {b}"),
                (Some(a), None) => format!("{name} >= {a}"),
                (None, Some(b)) => format!("{name} <= {b}"),
                (None, None) => continue,
            };
            out.push_str(&line);
            out.push('\n');
        }
        out.push_str(&format!("overlap {}\n", self.residue_overlap));
        out
    }

    /// Read filters written as one rule per line, returning the filter and a
    /// message for every line that could not be understood.
    ///
    /// ```text
    /// residues >= 6          # any metric name, or a per-residue property
    /// hydropathy <= 0        # such as rmsf or bfactor
    /// area 80 .. 2480        # both bounds at once
    /// overlap 0.6            # residues two cavities must share to merge
    /// ```
    ///
    /// Lines are rules, `#` starts a comment, and `>` / `<` are read as
    /// `>=` / `<=`. The result holds only the rules given, so an empty text
    /// means no filtering at all.
    pub fn parse_text(text: &str) -> (Self, Vec<String>) {
        let mut filter = Self::unfiltered();
        let mut problems = Vec::new();
        for (i, raw) in text.lines().enumerate() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            match parse_rule(line) {
                Ok(Rule::Overlap(v)) => filter.residue_overlap = v.clamp(0.05, 1.0),
                Ok(Rule::Criterion(c)) => filter = filter.with(c),
                Err(message) => problems.push(format!("line {}: {message}", i + 1)),
            }
        }
        (filter, problems)
    }
}

enum Rule {
    Overlap(f64),
    Criterion(Criterion),
}

/// Split a rule into a metric name, comparison operators and numbers.
fn tokenize(line: &str) -> Vec<String> {
    let mut spaced = String::with_capacity(line.len() + 8);
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '>' | '<' | '=' => {
                spaced.push(' ');
                spaced.push(c);
                // `>=` and `<=` are one operator.
                if chars.peek() == Some(&'=') {
                    chars.next();
                    spaced.push('=');
                }
                spaced.push(' ');
            }
            // `..` separates bounds; a single `.` is part of a number.
            '.' if chars.peek() == Some(&'.') => {
                chars.next();
                spaced.push_str(" .. ");
            }
            ',' => spaced.push(' '),
            _ => spaced.push(c),
        }
    }
    spaced.split_whitespace().map(str::to_string).collect()
}

fn parse_rule(line: &str) -> Result<Rule, String> {
    let tokens = tokenize(line);
    let (name, rest) = tokens.split_first().ok_or("empty rule")?;
    let number = |t: &String| t.parse::<f64>().map_err(|_| format!("'{t}' is not a number"));

    if matches!(name.to_lowercase().as_str(), "overlap" | "residue_overlap") {
        let value = rest
            .iter()
            .find_map(|t| t.parse::<f64>().ok())
            .ok_or("overlap needs a fraction, e.g. `overlap 0.6`")?;
        return Ok(Rule::Overlap(value));
    }

    let (mut min, mut max) = (None, None);
    if let Some(i) = rest.iter().position(|t| t == "..") {
        if let Some(t) = rest[..i].last() {
            min = Some(number(t)?);
        }
        if let Some(t) = rest[i + 1..].first() {
            max = Some(number(t)?);
        }
    } else {
        let mut i = 0;
        while i < rest.len() {
            let op = rest[i].as_str();
            let value = rest
                .get(i + 1)
                .ok_or_else(|| format!("'{op}' needs a number after it"))?;
            match op {
                ">" | ">=" => min = Some(number(value)?),
                "<" | "<=" => max = Some(number(value)?),
                "=" | "==" => {
                    let v = number(value)?;
                    min = Some(v);
                    max = Some(v);
                }
                other => {
                    return Err(format!("expected >=, <= or .., found '{other}'"));
                }
            }
            i += 2;
        }
    }
    if min.is_none() && max.is_none() {
        return Err("needs a bound, e.g. `volume >= 120` or `area 80 .. 2480`".into());
    }
    Ok(Rule::Criterion(Criterion { metric: Metric::parse(name), min, max }))
}

// ------------------------------------------------------------- statistics

/// How one metric is distributed over the cavities that were found, so a
/// threshold can be chosen from the data rather than guessed.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetricStats {
    pub label: String,
    /// Name to use in a filter rule.
    pub name: String,
    /// Cavities with a value for this metric.
    pub n: usize,
    /// Cavities where it could not be determined. A filter on this metric
    /// rejects those, so a large number here explains a surprising result.
    pub missing: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    /// 10th, 25th, 50th, 75th and 90th percentiles.
    pub percentiles: [f64; 5],
}

/// How many cavities one rule keeps, judged on its own.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CriterionReport {
    pub rule: String,
    pub passing: usize,
}

/// Everything pyKVFinder found, before and after filtering.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EnsembleStats {
    /// Every cavity detected, across all frames.
    pub cavities: usize,
    pub frames: usize,
    pub frames_with_cavities: usize,
    pub max_per_frame: usize,
    pub mean_per_frame: f64,
    pub metrics: Vec<MetricStats>,
    /// Each rule on its own, which shows up the one doing the damage.
    pub criteria: Vec<CriterionReport>,
    /// Cavities passing every rule at once.
    pub passing_all: usize,
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = fraction * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let weight = rank - lo as f64;
    sorted[lo] * (1.0 - weight) + sorted[hi] * weight
}

/// Distribution of one metric over `found`.
pub fn metric_stats(
    metric: &Metric,
    found: &[FrameCavity],
    properties: &ResidueProperties,
) -> MetricStats {
    let mut values: Vec<f64> = found
        .iter()
        .filter_map(|f| metric.of(&f.cavity, properties))
        .collect();
    values.sort_by(f64::total_cmp);
    let n = values.len();
    let mean = if n == 0 {
        f64::NAN
    } else {
        values.iter().sum::<f64>() / n as f64
    };
    MetricStats {
        label: metric.label(),
        name: metric.filter_name(),
        n,
        missing: found.len() - n,
        min: values.first().copied().unwrap_or(f64::NAN),
        max: values.last().copied().unwrap_or(f64::NAN),
        mean,
        percentiles: [
            percentile(&values, 0.10),
            percentile(&values, 0.25),
            percentile(&values, 0.50),
            percentile(&values, 0.75),
            percentile(&values, 0.90),
        ],
    }
}

/// Statistics over every cavity found, plus what each filter rule keeps.
///
/// Metrics cover the built-ins and every per-residue property supplied, so
/// the panel can list the names available in a filter.
pub fn ensemble_stats(
    found: &[FrameCavity],
    total_frames: usize,
    filter: &EnsembleFilter,
    properties: &ResidueProperties,
) -> EnsembleStats {
    let mut per_frame: HashMap<usize, usize> = HashMap::new();
    for f in found {
        *per_frame.entry(f.frame).or_default() += 1;
    }

    let metrics: Vec<MetricStats> = Metric::builtin()
        .into_iter()
        .chain(properties.keys().map(|p| Metric::ResidueMean(p.clone())))
        // Persistence belongs to a merged pocket, not a single cavity.
        .filter(|m| *m != Metric::Persistence)
        .map(|m| metric_stats(&m, found, properties))
        .collect();

    let criteria = filter
        .criteria
        .iter()
        .filter(|c| c.metric != Metric::Persistence)
        .map(|c| CriterionReport {
            rule: c.describe(),
            passing: found
                .iter()
                .filter(|f| {
                    c.metric
                        .of(&f.cavity, properties)
                        .is_some_and(|v| c.accepts(v))
                })
                .count(),
        })
        .collect();

    EnsembleStats {
        cavities: found.len(),
        frames: total_frames,
        frames_with_cavities: per_frame.len(),
        max_per_frame: per_frame.values().copied().max().unwrap_or(0),
        mean_per_frame: found.len() as f64 / total_frames.max(1) as f64,
        metrics,
        criteria,
        passing_all: found
            .iter()
            .filter(|f| filter.keeps_with(&f.cavity, properties))
            .count(),
    }
}

/// Read every per-frame result in a finished ensemble job.
///
/// Returns the cavities and the number of frames that were analysed. A frame
/// whose detection found nothing writes no results file but still counts, so
/// persistence is measured against the whole ensemble.
pub fn read_ensemble(job_dir: &Path) -> Result<(Vec<FrameCavity>, usize)> {
    let out = job_dir.join("out");
    let mut found = Vec::new();
    let mut frames = 0usize;
    for entry in std::fs::read_dir(&out).with_context(|| format!("reading {}", out.display()))? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(frame) = name.strip_prefix('f').and_then(|n| n.parse::<usize>().ok()) else {
            continue;
        };
        frames += 1;
        let Ok(text) = std::fs::read_to_string(entry.path().join("frame.KVFinder.results.toml"))
        else {
            continue;
        };
        for cavity in parse_results(&text)? {
            found.push(FrameCavity { frame, cavity });
        }
    }
    found.sort_by(|a, b| {
        a.frame
            .cmp(&b.frame)
            .then_with(|| a.cavity.id.cmp(&b.cavity.id))
    });
    Ok((found, frames))
}

/// A cavity together with the frame it came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FrameCavity {
    pub frame: usize,
    pub cavity: Cavity,
}

/// A pocket seen across several frames of an ensemble.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConsensusPocket {
    /// 1-based rank by persistence then volume.
    pub rank: usize,
    /// Fraction of frames in which this pocket was detected (0..=1).
    pub persistence: f64,
    pub frames: Vec<usize>,
    /// Frame with the largest instance; the one worth looking at.
    pub best_frame: usize,
    pub best_cavity_id: String,
    pub max_volume: f64,
    pub mean_volume: f64,
    pub mean_area: f64,
    pub mean_hydropathy: Option<f64>,
    pub max_depth: Option<f64>,
    /// Residues lining it in any frame, most frequent first.
    pub residues: Vec<LiningResidue>,
    /// How many frames each residue lines it in.
    pub residue_frequency: Vec<usize>,
}

fn residue_key(r: &LiningResidue) -> (String, String) {
    (r.chain.clone(), r.number.clone())
}

/// Lining residues as sorted, deduplicated interned ids.
///
/// Merging compares every cavity against each group's representative, and an
/// unfiltered ensemble can hold tens of thousands of cavities, so the
/// comparison has to be a scan over integers rather than string hashing.
fn interned(c: &Cavity, ids: &mut HashMap<(String, String), u32>) -> Vec<u32> {
    let mut v: Vec<u32> = c
        .residues
        .iter()
        .map(|r| {
            let next = ids.len() as u32;
            *ids.entry(residue_key(r)).or_insert(next)
        })
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// Fraction of the smaller residue set that two cavities share. Both inputs
/// must be sorted and deduplicated.
fn overlap(a: &[u32], b: &[u32]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (mut i, mut j, mut shared) = (0usize, 0usize, 0usize);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                shared += 1;
                i += 1;
                j += 1;
            }
        }
    }
    shared as f64 / a.len().min(b.len()) as f64
}

/// Merge per-frame cavities into consensus pockets.
///
/// Cavities are taken largest first; each joins the first pocket whose
/// representative shares at least `residue_overlap` of its lining residues,
/// otherwise it starts a new pocket. `total_frames` is the denominator for
/// persistence, so frames in which nothing was found still count.
pub fn consensus_pockets(
    found: &[FrameCavity],
    total_frames: usize,
    filter: &EnsembleFilter,
) -> Vec<ConsensusPocket> {
    consensus_pockets_with(found, total_frames, filter, &ResidueProperties::new())
}

/// As [`consensus_pockets`], with per-residue properties available to filters
/// such as `rmsf`.
pub fn consensus_pockets_with(
    found: &[FrameCavity],
    total_frames: usize,
    filter: &EnsembleFilter,
    properties: &ResidueProperties,
) -> Vec<ConsensusPocket> {
    let mut kept: Vec<&FrameCavity> = found
        .iter()
        .filter(|f| filter.keeps_with(&f.cavity, properties))
        .collect();
    kept.sort_by(|a, b| b.cavity.volume.total_cmp(&a.cavity.volume));

    // groups[i] = (representative residue ids, members)
    let mut ids: HashMap<(String, String), u32> = HashMap::new();
    let mut groups: Vec<(Vec<u32>, Vec<&FrameCavity>)> = Vec::new();
    for item in kept {
        let keys = interned(&item.cavity, &mut ids);
        match groups
            .iter_mut()
            .find(|(rep, _)| overlap(rep, &keys) >= filter.residue_overlap)
        {
            Some((_, members)) => members.push(item),
            None => groups.push((keys, vec![item])),
        }
    }

    let denominator = total_frames.max(1) as f64;
    let mut pockets: Vec<ConsensusPocket> = groups
        .into_iter()
        .map(|(_, members)| {
            let mut frames: Vec<usize> = members.iter().map(|m| m.frame).collect();
            frames.sort_unstable();
            frames.dedup();
            let best = members
                .iter()
                .max_by(|a, b| a.cavity.volume.total_cmp(&b.cavity.volume))
                .expect("groups are never empty");
            let n = members.len() as f64;

            // Residue frequency across the frames this pocket appears in.
            let mut counts: std::collections::HashMap<(String, String), (LiningResidue, usize)> =
                std::collections::HashMap::new();
            for m in &members {
                for r in &m.cavity.residues {
                    counts
                        .entry(residue_key(r))
                        .or_insert_with(|| (r.clone(), 0))
                        .1 += 1;
                }
            }
            let mut by_frequency: Vec<(LiningResidue, usize)> = counts.into_values().collect();
            by_frequency.sort_by(|a, b| {
                b.1.cmp(&a.1)
                    .then_with(|| a.0.chain.cmp(&b.0.chain))
                    .then_with(|| {
                        a.0.number
                            .parse::<i64>()
                            .unwrap_or(0)
                            .cmp(&b.0.number.parse::<i64>().unwrap_or(0))
                    })
            });

            let hydropathies: Vec<f64> = members.iter().filter_map(|m| m.cavity.avg_hydropathy).collect();
            ConsensusPocket {
                rank: 0,
                persistence: frames.len() as f64 / denominator,
                frames,
                best_frame: best.frame,
                best_cavity_id: best.cavity.id.clone(),
                max_volume: best.cavity.volume,
                mean_volume: members.iter().map(|m| m.cavity.volume).sum::<f64>() / n,
                mean_area: members.iter().map(|m| m.cavity.area).sum::<f64>() / n,
                mean_hydropathy: (!hydropathies.is_empty())
                    .then(|| hydropathies.iter().sum::<f64>() / hydropathies.len() as f64),
                max_depth: members.iter().filter_map(|m| m.cavity.max_depth).reduce(f64::max),
                residues: by_frequency.iter().map(|(r, _)| r.clone()).collect(),
                residue_frequency: by_frequency.iter().map(|(_, n)| *n).collect(),
            }
        })
        .collect();

    // Criteria that only make sense once frames are merged.
    pockets.retain(|p| filter.keeps_pocket(p));
    // Persistent first, then big.
    pockets.sort_by(|a, b| {
        b.persistence
            .total_cmp(&a.persistence)
            .then_with(|| b.max_volume.total_cmp(&a.max_volume))
    });
    for (i, p) in pockets.iter_mut().enumerate() {
        p.rank = i + 1;
    }
    pockets
}

impl ConsensusPocket {
    /// Residues lining it in at least `fraction` of its own frames.
    pub fn core_residues(&self, fraction: f64) -> Vec<&LiningResidue> {
        let threshold = (self.frames.len() as f64 * fraction).ceil() as usize;
        self.residues
            .iter()
            .zip(&self.residue_frequency)
            .filter(|(_, n)| **n >= threshold.max(1))
            .map(|(r, _)| r)
            .collect()
    }

    pub fn residue_labels(&self, limit: usize) -> String {
        let all = self.core_residues(0.5);
        let shown: Vec<String> = all
            .iter()
            .take(limit)
            .map(|r| format!("{}:{} {}", r.chain, r.name, r.number))
            .collect();
        let mut text = shown.join(", ");
        if all.len() > limit {
            text.push_str(&format!(", … (+{})", all.len() - limit));
        }
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_1hsg_results() {
        let text = include_str!("../tests/fixtures/receptor.KVFinder.results.toml");
        let cavities = parse_results(text).unwrap();
        assert_eq!(cavities.len(), 2);
        let kaa = &cavities[0];
        assert_eq!(kaa.id, "KAA");
        assert!((kaa.volume - 689.69).abs() < 1e-6);
        assert_eq!(kaa.max_depth, Some(6.63));
        assert_eq!(kaa.avg_hydropathy, Some(-0.52));
        // The HIV protease active site: catalytic Asp25 of both chains.
        assert!(kaa.residues.iter().any(|r| r.chain == "A" && r.number == "25" && r.name == "ASP"));
        assert!(kaa.residues.iter().any(|r| r.chain == "B" && r.number == "25" && r.name == "ASP"));
    }

    #[test]
    fn sanitizes_cavity_lines() {
        let line = "ATOM      2  HA  KAA   259       8.830  20.686   1.811 -0.12  1.80";
        let fixed = sanitize_cavity_pdb(line);
        let fixed = fixed.trim_end_matches('\n');
        assert_eq!(&fixed[..22], &line[..22]);
        assert_eq!(fixed[22..26].trim(), "1");
        assert_eq!(&fixed[26..54], &line[26..54]);
        assert_eq!(&fixed[54..60], "  1.00");
        assert_eq!(fixed[60..66].trim(), "1.80");
        assert_eq!(fixed.len(), 78);
        assert_eq!(&fixed[76..78], " H");

        let two = "ATOM      1  HA  KAA   259       8.830  20.686   1.811 -0.12  1.80
ATOM      2  H   KAB   259       1.000   2.000   3.000  0.50  0.00";
        let fixed = sanitize_cavity_pdb(two);
        let numbers: Vec<&str> = fixed.lines().map(|l| l[22..26].trim()).collect();
        assert_eq!(numbers, ["1", "2"]);
    }

    fn residues(chain: &str, numbers: &[i32]) -> Vec<LiningResidue> {
        numbers
            .iter()
            .map(|n| LiningResidue {
                number: n.to_string(),
                chain: chain.into(),
                name: "ALA".into(),
            })
            .collect()
    }

    fn cavity(id: &str, volume: f64, res: Vec<LiningResidue>) -> Cavity {
        Cavity {
            id: id.into(),
            volume,
            area: 300.0,
            max_depth: Some(5.0),
            avg_depth: Some(2.0),
            avg_hydropathy: Some(-0.4),
            residues: res,
        }
    }

    #[test]
    fn merges_pockets_across_frames_by_residue_overlap() {
        // The same pocket in frames 0,1,2 (residues drift a little), plus a
        // different one that appears once.
        let found = vec![
            FrameCavity { frame: 0, cavity: cavity("KAA", 400.0, residues("A", &[10, 11, 12, 13, 14, 15])) },
            FrameCavity { frame: 1, cavity: cavity("KAA", 500.0, residues("A", &[11, 12, 13, 14, 15, 16])) },
            FrameCavity { frame: 2, cavity: cavity("KAB", 300.0, residues("A", &[10, 12, 13, 14, 15, 17])) },
            FrameCavity { frame: 1, cavity: cavity("KAB", 350.0, residues("B", &[70, 71, 72, 73, 74, 75])) },
        ];
        let pockets = consensus_pockets(&found, 4, &EnsembleFilter::default());
        assert_eq!(pockets.len(), 2);

        let main = &pockets[0];
        assert_eq!(main.rank, 1);
        assert_eq!(main.frames, vec![0, 1, 2]);
        assert_eq!(main.persistence, 0.75, "3 of 4 frames");
        assert_eq!(main.best_frame, 1, "largest instance");
        assert_eq!(main.max_volume, 500.0);
        // Residues 12-15 line it in all three frames.
        assert_eq!(main.core_residues(1.0).len(), 4);

        let rare = &pockets[1];
        assert_eq!(rare.frames, vec![1]);
        assert_eq!(rare.persistence, 0.25);
    }

    #[test]
    fn applies_the_paper_filters() {
        let f = EnsembleFilter::default();
        assert!(f.keeps(&cavity("KAA", 400.0, residues("A", &[1, 2, 3, 4, 5, 6]))));
        // Too few residues (needs more than 5).
        assert!(!f.keeps(&cavity("KAA", 400.0, residues("A", &[1, 2, 3, 4, 5]))));
        // Too small.
        assert!(!f.keeps(&cavity("KAA", 100.0, residues("A", &[1, 2, 3, 4, 5, 6]))));
        // Hydrophobic cavities are dropped by the hydropathy < 0 filter.
        let mut hydrophobic = cavity("KAA", 400.0, residues("A", &[1, 2, 3, 4, 5, 6]));
        hydrophobic.avg_hydropathy = Some(0.5);
        assert!(!f.keeps(&hydrophobic));
    }

    #[test]
    fn filters_on_md_derived_residue_properties() {
        // Flexible loop residues 1-6, rigid core residues 10-15.
        let mut rmsf = HashMap::new();
        for n in 1..=6 {
            rmsf.insert(("A".to_string(), n.to_string()), 8.0);
        }
        for n in 10..=15 {
            rmsf.insert(("A".to_string(), n.to_string()), 0.9);
        }
        let mut properties = ResidueProperties::new();
        properties.insert("rmsf".into(), rmsf);

        let filter = EnsembleFilter::default()
            .with(Criterion::at_most(Metric::ResidueMean("rmsf".into()), 2.0));
        let floppy = cavity("KAA", 400.0, residues("A", &[1, 2, 3, 4, 5, 6]));
        let rigid = cavity("KAB", 400.0, residues("A", &[10, 11, 12, 13, 14, 15]));
        assert!(!filter.keeps_with(&floppy, &properties), "flexible pocket should be dropped");
        assert!(filter.keeps_with(&rigid, &properties), "rigid pocket should be kept");

        // An unknown property rejects rather than silently passing.
        let unknown = EnsembleFilter::default()
            .with(Criterion::at_most(Metric::ResidueMean("sasa".into()), 1.0));
        assert!(!unknown.keeps_with(&rigid, &properties));
    }

    #[test]
    fn parses_metric_names_and_filters_on_persistence() {
        assert_eq!(Metric::parse("Volume"), Metric::Volume);
        assert_eq!(Metric::parse("max_depth"), Metric::MaxDepth);
        assert_eq!(Metric::parse("mean:rmsf"), Metric::ResidueMean("rmsf".into()));
        assert_eq!(Metric::parse("bfactor"), Metric::ResidueMean("bfactor".into()));

        let found = vec![
            FrameCavity { frame: 0, cavity: cavity("KAA", 400.0, residues("A", &[1, 2, 3, 4, 5, 6])) },
            FrameCavity { frame: 1, cavity: cavity("KAA", 400.0, residues("A", &[1, 2, 3, 4, 5, 6])) },
            FrameCavity { frame: 2, cavity: cavity("KAB", 300.0, residues("B", &[20, 21, 22, 23, 24, 25])) },
        ];
        // Keep only pockets present in at least half the frames.
        let filter = EnsembleFilter::default()
            .with(Criterion::at_least(Metric::Persistence, 0.5));
        let pockets = consensus_pockets(&found, 4, &filter);
        assert_eq!(pockets.len(), 1);
        assert_eq!(pockets[0].persistence, 0.5);
    }

    #[test]
    fn reads_filters_written_as_text() {
        let (filter, problems) = EnsembleFilter::parse_text(
            "residues >= 6
             hydropathy <= 0     # hydrophilic only
             area 80 .. 2480
             volume > 120
             rmsf <= 1.5
             # a comment line
             overlap 0.75
",
        );
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(filter.residue_overlap, 0.75);
        assert_eq!(filter.criteria.len(), 5);
        assert_eq!(
            filter.criteria[0],
            Criterion::at_least(Metric::ResidueCount, 6.0)
        );
        assert_eq!(filter.criteria[1], Criterion::at_most(Metric::Hydropathy, 0.0));
        assert_eq!(filter.criteria[2], Criterion::between(Metric::Area, 80.0, 2480.0));
        // `>` is read as `>=`.
        assert_eq!(filter.criteria[3], Criterion::at_least(Metric::Volume, 120.0));
        // Anything else is a per-residue property.
        assert_eq!(
            filter.criteria[4],
            Criterion::at_most(Metric::ResidueMean("rmsf".into()), 1.5)
        );
    }

    #[test]
    fn filter_text_round_trips() {
        let original = EnsembleFilter::default()
            .with(Criterion::at_most(Metric::ResidueMean("rmsf".into()), 2.0))
            .with(Criterion::at_least(Metric::Persistence, 0.3));
        let (parsed, problems) = EnsembleFilter::parse_text(&original.to_text());
        assert!(problems.is_empty(), "{problems:?}");
        assert_eq!(parsed, original);
    }

    #[test]
    fn empty_filter_text_keeps_every_cavity() {
        let (filter, problems) = EnsembleFilter::parse_text("  
# nothing
");
        assert!(problems.is_empty());
        assert!(filter.criteria.is_empty());
        // Even a cavity the paper filters would drop.
        let mut tiny = cavity("KAA", 1.0, residues("A", &[1]));
        tiny.avg_hydropathy = Some(2.0);
        assert!(filter.keeps(&tiny));
    }

    #[test]
    fn reports_bad_filter_lines_without_losing_the_good_ones() {
        let (filter, problems) = EnsembleFilter::parse_text("volume >= 120
area wobbly
");
        assert_eq!(filter.criteria.len(), 1);
        assert_eq!(problems.len(), 1);
        assert!(problems[0].starts_with("line 2:"), "{problems:?}");
    }

    #[test]
    fn describes_what_was_found_and_what_each_rule_keeps() {
        // Three cavities: one passes everything, one is too small, one is
        // hydrophobic.
        let mut hydrophobic = cavity("KAC", 400.0, residues("A", &[1, 2, 3, 4, 5, 6]));
        hydrophobic.avg_hydropathy = Some(0.8);
        let found = vec![
            FrameCavity { frame: 0, cavity: cavity("KAA", 400.0, residues("A", &[1, 2, 3, 4, 5, 6])) },
            FrameCavity { frame: 0, cavity: cavity("KAB", 10.0, residues("A", &[20, 21, 22, 23, 24, 25])) },
            FrameCavity { frame: 1, cavity: hydrophobic },
        ];
        let filter = EnsembleFilter::default();
        let stats = ensemble_stats(&found, 4, &filter, &ResidueProperties::new());

        assert_eq!(stats.cavities, 3);
        assert_eq!(stats.frames, 4);
        assert_eq!(stats.frames_with_cavities, 2, "two frames held a cavity");
        assert_eq!(stats.max_per_frame, 2);
        assert_eq!(stats.passing_all, 1);

        let volume = stats.metrics.iter().find(|m| m.name == "volume").unwrap();
        assert_eq!(volume.n, 3);
        assert_eq!(volume.missing, 0);
        assert_eq!(volume.min, 10.0);
        assert_eq!(volume.max, 400.0);
        assert_eq!(volume.percentiles[2], 400.0, "median of 10, 400, 400");

        // Each rule judged on its own, which shows which one is strict.
        let by_rule = |needle: &str| {
            stats
                .criteria
                .iter()
                .find(|c| c.rule.contains(needle))
                .unwrap_or_else(|| panic!("no rule for {needle}: {:?}", stats.criteria))
                .passing
        };
        assert_eq!(by_rule("residues"), 3, "all have 6 lining residues");
        assert_eq!(by_rule("hydropathy"), 2, "one is hydrophobic");
        assert_eq!(by_rule("volume"), 2, "one is too small");

        // A metric nothing carries is reported as missing, not as zero.
        let stats = ensemble_stats(&found, 4, &filter, &ResidueProperties::new());
        let rmsf = metric_stats(
            &Metric::ResidueMean("rmsf".into()),
            &found,
            &ResidueProperties::new(),
        );
        assert_eq!(rmsf.n, 0);
        assert_eq!(rmsf.missing, 3);
        assert!(!stats.metrics.iter().any(|m| m.name == "rmsf"), "not supplied");
    }

    #[test]
    fn builds_the_ensemble_loop() {
        let cmd = ensemble_command(&Settings::default());
        assert!(cmd.contains("for f in frames/*.pdb"));
        assert!(cmd.contains("-b frame"));
        assert!(cmd.contains("-O \"out/$n\""));
        assert_eq!(frame_results(7), "out/f00007/frame.KVFinder.results.toml");
    }

    #[test]
    fn builds_command() {
        let cmd = command(&Settings { ligand_adjust: true, ..Default::default() });
        assert!(cmd.starts_with("pyKVFinder receptor.pdb -D --hydropathy -O out"));
        assert!(cmd.ends_with("-L ligand.pdb"));
    }
}
