pub mod parse;
pub mod traverse;

use serde_with::skip_serializing_none;
pub use traverse::*;

use rig::Embed;

#[skip_serializing_none]
#[derive(Embed, serde::Deserialize, serde::Serialize, Debug, Clone)]
pub struct CodeSnippet {
    pub path: String,
    pub class: Option<String>,
    pub name: String,
    pub body: String,

    #[embed]
    pub summary: String,
}

#[cfg(test)]
#[path = "../tests/utils/mod.rs"]
mod test_utils;
