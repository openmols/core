//! Discovery-campaign projects.
//!
//! A project is a self-contained folder:
//!
//! ```text
//! my-campaign/
//!   project.db        SQLite: metadata, structures, views, conversations, notes
//!   structures/       copies of every structure added to the project
//! ```
//!
//! The crate is UI-agnostic: view state and chat messages are stored as JSON
//! documents whose schema belongs to the application.

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const DB_FILE: &str = "project.db";
const STRUCTURES_DIR: &str = "structures";
const JOBS_DIR: &str = "jobs";
const SCHEMA_VERSION: i64 = 1;

pub type Id = i64;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct StructureEntry {
    pub id: Id,
    pub name: String,
    /// Path relative to the project root.
    pub file: String,
    /// Where it came from, e.g. `rcsb:1HSG` or the original path.
    pub source: Option<String>,
    pub added_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ViewEntry {
    pub id: Id,
    pub name: String,
    pub structure_id: Option<Id>,
    /// Application-defined JSON (camera, representations, selection…).
    pub state: serde_json::Value,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Conversation {
    pub id: Id,
    pub title: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Job {
    pub id: Id,
    /// Tool id, e.g. `pykvfinder`.
    pub tool: String,
    /// `queued`, `running`, `done`, `failed`, `cancelled` or `interrupted`.
    pub status: String,
    /// Inputs/settings (tool-defined JSON).
    pub params: serde_json::Value,
    /// Parsed outputs (tool-defined JSON), once finished.
    pub result: Option<serde_json::Value>,
    pub created_at: i64,
    pub finished_at: Option<i64>,
}

impl Job {
    pub fn is_active(&self) -> bool {
        matches!(self.status.as_str(), "queued" | "running")
    }

    /// Folder of this job relative to the project root.
    pub fn dir(&self) -> String {
        format!("{JOBS_DIR}/{}", self.id)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Note {
    pub id: Id,
    pub title: String,
    pub body: String,
    pub created_at: i64,
}

pub struct Project {
    root: PathBuf,
    db: Connection,
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

impl Project {
    /// Create a new project in `root` (created if missing; must not already be a project).
    pub fn create(root: impl AsRef<Path>, name: &str) -> Result<Self> {
        let root = root.as_ref();
        if root.join(DB_FILE).exists() {
            bail!("{} already contains a project", root.display());
        }
        std::fs::create_dir_all(root.join(STRUCTURES_DIR))
            .with_context(|| format!("creating {}", root.display()))?;
        let project = Self::connect(root)?;
        project.set_meta("name", name)?;
        project.set_meta("created_at", &now().to_string())?;
        Ok(project)
    }

    /// Open an existing project folder.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref();
        if !root.join(DB_FILE).exists() {
            bail!("{} is not an openmols project (no {DB_FILE})", root.display());
        }
        Self::connect(root)
    }

    pub fn is_project(root: impl AsRef<Path>) -> bool {
        root.as_ref().join(DB_FILE).exists()
    }

    fn connect(root: &Path) -> Result<Self> {
        let db = Connection::open(root.join(DB_FILE)).context("opening project database")?;
        db.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             CREATE TABLE IF NOT EXISTS meta(
                 key TEXT PRIMARY KEY,
                 value TEXT NOT NULL);
             CREATE TABLE IF NOT EXISTS structures(
                 id INTEGER PRIMARY KEY,
                 name TEXT NOT NULL,
                 file TEXT NOT NULL,
                 source TEXT,
                 added_at INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS views(
                 id INTEGER PRIMARY KEY,
                 name TEXT NOT NULL,
                 structure_id INTEGER REFERENCES structures(id) ON DELETE CASCADE,
                 state TEXT NOT NULL,
                 created_at INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS conversations(
                 id INTEGER PRIMARY KEY,
                 title TEXT NOT NULL,
                 created_at INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS messages(
                 id INTEGER PRIMARY KEY,
                 conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                 body TEXT NOT NULL,
                 created_at INTEGER NOT NULL);
             CREATE TABLE IF NOT EXISTS jobs(
                 id INTEGER PRIMARY KEY,
                 tool TEXT NOT NULL,
                 status TEXT NOT NULL,
                 params TEXT NOT NULL,
                 result TEXT,
                 created_at INTEGER NOT NULL,
                 finished_at INTEGER);
             CREATE TABLE IF NOT EXISTS notes(
                 id INTEGER PRIMARY KEY,
                 title TEXT NOT NULL,
                 body TEXT NOT NULL,
                 created_at INTEGER NOT NULL);",
        )?;
        let project = Self {
            root: root.to_path_buf(),
            db,
        };
        project.set_meta("schema_version", &SCHEMA_VERSION.to_string())?;
        // Jobs that were running when the app last closed can't be resumed.
        project.db.execute(
            "UPDATE jobs SET status = 'interrupted', finished_at = ?1 WHERE status IN ('queued', 'running')",
            [now()],
        )?;
        Ok(project)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn name(&self) -> String {
        self.meta("name").ok().flatten().unwrap_or_else(|| {
            self.root
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        })
    }

    pub fn meta(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .db
            .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
            .optional()?)
    }

    pub fn set_meta(&self, key: &str, value: &str) -> Result<()> {
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    // ---------------------------------------------------------- structures

    /// Copy a structure file into the project and register it.
    pub fn add_structure_file(&self, path: impl AsRef<Path>, name: Option<&str>) -> Result<StructureEntry> {
        let path = path.as_ref();
        let file_name = path
            .file_name()
            .context("structure path has no file name")?
            .to_string_lossy()
            .into_owned();
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        let name = name.map(str::to_string).unwrap_or_else(|| stem(&file_name));
        self.add_structure_bytes(&file_name, &bytes, &name, Some(&path.to_string_lossy()))
    }

    /// Store structure data (e.g. downloaded from the PDB) in the project.
    pub fn add_structure_bytes(
        &self,
        file_name: &str,
        bytes: &[u8],
        name: &str,
        source: Option<&str>,
    ) -> Result<StructureEntry> {
        let dir = self.root.join(STRUCTURES_DIR);
        std::fs::create_dir_all(&dir)?;
        let target = unique_path(&dir, file_name);
        std::fs::write(&target, bytes).with_context(|| format!("writing {}", target.display()))?;
        let file = format!(
            "{STRUCTURES_DIR}/{}",
            target.file_name().unwrap().to_string_lossy()
        );
        let added_at = now();
        self.db.execute(
            "INSERT INTO structures(name, file, source, added_at) VALUES (?1, ?2, ?3, ?4)",
            params![name, file, source, added_at],
        )?;
        Ok(StructureEntry {
            id: self.db.last_insert_rowid(),
            name: name.to_string(),
            file,
            source: source.map(str::to_string),
            added_at,
        })
    }

    pub fn structures(&self) -> Result<Vec<StructureEntry>> {
        let mut stmt = self
            .db
            .prepare("SELECT id, name, file, source, added_at FROM structures ORDER BY id")?;
        let rows = stmt.query_map([], |r| {
            Ok(StructureEntry {
                id: r.get(0)?,
                name: r.get(1)?,
                file: r.get(2)?,
                source: r.get(3)?,
                added_at: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn structure(&self, id: Id) -> Result<Option<StructureEntry>> {
        Ok(self.structures()?.into_iter().find(|s| s.id == id))
    }

    pub fn structure_path(&self, entry: &StructureEntry) -> PathBuf {
        self.root.join(&entry.file)
    }

    pub fn remove_structure(&self, id: Id) -> Result<()> {
        if let Some(entry) = self.structure(id)? {
            std::fs::remove_file(self.structure_path(&entry)).ok();
        }
        self.db.execute("DELETE FROM structures WHERE id = ?1", [id])?;
        Ok(())
    }

    // --------------------------------------------------------------- views

    /// Save a named view; an existing view with the same name is replaced.
    pub fn save_view(&self, name: &str, structure_id: Option<Id>, state: &serde_json::Value) -> Result<ViewEntry> {
        self.db.execute("DELETE FROM views WHERE name = ?1", [name])?;
        let created_at = now();
        self.db.execute(
            "INSERT INTO views(name, structure_id, state, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![name, structure_id, state.to_string(), created_at],
        )?;
        Ok(ViewEntry {
            id: self.db.last_insert_rowid(),
            name: name.to_string(),
            structure_id,
            state: state.clone(),
            created_at,
        })
    }

    pub fn views(&self) -> Result<Vec<ViewEntry>> {
        let mut stmt = self
            .db
            .prepare("SELECT id, name, structure_id, state, created_at FROM views ORDER BY id")?;
        let rows = stmt.query_map([], |r| {
            let state: String = r.get(3)?;
            Ok(ViewEntry {
                id: r.get(0)?,
                name: r.get(1)?,
                structure_id: r.get(2)?,
                state: serde_json::from_str(&state).unwrap_or(serde_json::Value::Null),
                created_at: r.get(4)?,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn delete_view(&self, id: Id) -> Result<()> {
        self.db.execute("DELETE FROM views WHERE id = ?1", [id])?;
        Ok(())
    }

    // ------------------------------------------------------- conversations

    pub fn create_conversation(&self, title: &str) -> Result<Conversation> {
        let created_at = now();
        self.db.execute(
            "INSERT INTO conversations(title, created_at) VALUES (?1, ?2)",
            params![title, created_at],
        )?;
        Ok(Conversation {
            id: self.db.last_insert_rowid(),
            title: title.to_string(),
            created_at,
        })
    }

    pub fn conversations(&self) -> Result<Vec<Conversation>> {
        let mut stmt = self
            .db
            .prepare("SELECT id, title, created_at FROM conversations ORDER BY id")?;
        let rows = stmt.query_map([], |r| {
            Ok(Conversation {
                id: r.get(0)?,
                title: r.get(1)?,
                created_at: r.get(2)?,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn rename_conversation(&self, id: Id, title: &str) -> Result<()> {
        self.db
            .execute("UPDATE conversations SET title = ?2 WHERE id = ?1", params![id, title])?;
        Ok(())
    }

    /// Append a message (any JSON document) to a conversation.
    pub fn append_message(&self, conversation: Id, message: &serde_json::Value) -> Result<()> {
        self.db.execute(
            "INSERT INTO messages(conversation_id, body, created_at) VALUES (?1, ?2, ?3)",
            params![conversation, message.to_string(), now()],
        )?;
        Ok(())
    }

    pub fn messages(&self, conversation: Id) -> Result<Vec<serde_json::Value>> {
        let mut stmt = self
            .db
            .prepare("SELECT body FROM messages WHERE conversation_id = ?1 ORDER BY id")?;
        let rows = stmt.query_map([conversation], |r| r.get::<_, String>(0))?;
        rows.map(|r| Ok(serde_json::from_str(&r?)?)).collect()
    }

    // ---------------------------------------------------------------- jobs

    /// Record a new job (status `queued`) and create its folder.
    pub fn create_job(&self, tool: &str, params: &serde_json::Value) -> Result<Job> {
        let created_at = now();
        self.db.execute(
            "INSERT INTO jobs(tool, status, params, created_at) VALUES (?1, 'queued', ?2, ?3)",
            params![tool, params.to_string(), created_at],
        )?;
        let job = Job {
            id: self.db.last_insert_rowid(),
            tool: tool.to_string(),
            status: "queued".into(),
            params: params.clone(),
            result: None,
            created_at,
            finished_at: None,
        };
        std::fs::create_dir_all(self.root.join(job.dir()))?;
        Ok(job)
    }

    pub fn set_job_status(&self, id: Id, status: &str, result: Option<&serde_json::Value>) -> Result<()> {
        let finished = (!matches!(status, "queued" | "running")).then(now);
        self.db.execute(
            "UPDATE jobs SET status = ?2, result = COALESCE(?3, result), finished_at = ?4 WHERE id = ?1",
            params![id, status, result.map(|r| r.to_string()), finished],
        )?;
        Ok(())
    }

    /// Most recent first.
    pub fn jobs(&self) -> Result<Vec<Job>> {
        let mut stmt = self.db.prepare(
            "SELECT id, tool, status, params, result, created_at, finished_at FROM jobs ORDER BY id DESC",
        )?;
        let rows = stmt.query_map([], |r| {
            let params: String = r.get(3)?;
            let result: Option<String> = r.get(4)?;
            Ok(Job {
                id: r.get(0)?,
                tool: r.get(1)?,
                status: r.get(2)?,
                params: serde_json::from_str(&params).unwrap_or(serde_json::Value::Null),
                result: result.and_then(|r| serde_json::from_str(&r).ok()),
                created_at: r.get(5)?,
                finished_at: r.get(6)?,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }

    pub fn job(&self, id: Id) -> Result<Option<Job>> {
        Ok(self.jobs()?.into_iter().find(|j| j.id == id))
    }

    pub fn job_path(&self, job: &Job) -> PathBuf {
        self.root.join(job.dir())
    }

    // --------------------------------------------------------------- notes

    pub fn add_note(&self, title: &str, body: &str) -> Result<Note> {
        let created_at = now();
        self.db.execute(
            "INSERT INTO notes(title, body, created_at) VALUES (?1, ?2, ?3)",
            params![title, body, created_at],
        )?;
        Ok(Note {
            id: self.db.last_insert_rowid(),
            title: title.to_string(),
            body: body.to_string(),
            created_at,
        })
    }

    pub fn notes(&self) -> Result<Vec<Note>> {
        let mut stmt = self
            .db
            .prepare("SELECT id, title, body, created_at FROM notes ORDER BY id")?;
        let rows = stmt.query_map([], |r| {
            Ok(Note {
                id: r.get(0)?,
                title: r.get(1)?,
                body: r.get(2)?,
                created_at: r.get(3)?,
            })
        })?;
        Ok(rows.collect::<Result<_, _>>()?)
    }
}

fn stem(file_name: &str) -> String {
    let lower = file_name.to_lowercase();
    let trimmed = lower.strip_suffix(".gz").map(|_| &file_name[..file_name.len() - 3]).unwrap_or(file_name);
    Path::new(trimmed)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| trimmed.to_string())
}

/// `dir/name`, or `dir/name-2.ext`, `-3`… if taken.
fn unique_path(dir: &Path, file_name: &str) -> PathBuf {
    let candidate = dir.join(file_name);
    if !candidate.exists() {
        return candidate;
    }
    let (base, ext) = match file_name.split_once('.') {
        Some((b, e)) => (b.to_string(), format!(".{e}")),
        None => (file_name.to_string(), String::new()),
    };
    (2..)
        .map(|i| dir.join(format!("{base}-{i}{ext}")))
        .find(|p| !p.exists())
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("campaign");
        let p = Project::create(&root, "HIV protease").unwrap();
        assert_eq!(p.name(), "HIV protease");

        let s = p.add_structure_bytes("1hsg.pdb", b"ATOM", "1HSG", Some("rcsb:1HSG")).unwrap();
        let s2 = p.add_structure_bytes("1hsg.pdb", b"ATOM", "1HSG copy", None).unwrap();
        assert_eq!(s2.file, "structures/1hsg-2.pdb");
        assert!(p.structure_path(&s).exists());

        p.save_view("pocket", Some(s.id), &json!({"distance": 12.5})).unwrap();
        p.save_view("pocket", Some(s.id), &json!({"distance": 20.0})).unwrap();
        let views = p.views().unwrap();
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].state["distance"], 20.0);

        let c = p.create_conversation("Binding site").unwrap();
        p.append_message(c.id, &json!({"role": "user", "content": "hi"})).unwrap();
        drop(p);

        let p = Project::open(&root).unwrap();
        assert_eq!(p.structures().unwrap().len(), 2);
        assert_eq!(p.messages(c.id).unwrap()[0]["content"], "hi");
        p.remove_structure(s.id).unwrap();
        assert!(p.views().unwrap().is_empty(), "views cascade with their structure");
        assert!(Project::create(&root, "again").is_err());

        let job = p.create_job("pykvfinder", &json!({"entity": "1hsg"})).unwrap();
        assert!(p.job_path(&job).is_dir());
        p.set_job_status(job.id, "running", None).unwrap();
        drop(p);
        let p = Project::open(&root).unwrap();
        assert_eq!(p.job(job.id).unwrap().unwrap().status, "interrupted");
        p.set_job_status(job.id, "done", Some(&json!({"cavities": 2}))).unwrap();
        let j = p.job(job.id).unwrap().unwrap();
        assert_eq!(j.result.unwrap()["cavities"], 2);
        assert!(j.finished_at.is_some());
    }
}
