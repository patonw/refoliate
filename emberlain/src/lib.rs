pub mod parse;
pub mod traverse;

pub use traverse::*;

use rig::Embed;

#[derive(Embed, serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct CodeSnippet {
    pub path: String,
    pub name: String,
    pub body: String,

    #[embed]
    pub summary: String,
}

#[cfg(test)]
#[path = "../tests/utils/mod.rs"]
mod test_utils;
