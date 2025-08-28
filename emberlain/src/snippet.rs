use std::borrow::Cow;

use anyhow::Result;
use cached::proc_macro::cached;
use rig::Embed;
use serde_with::{serde_as, skip_serializing_none};
use uuid::Uuid;

#[cached]
fn make_id_hash(
    path: String,
    interface: Option<String>,
    class: Option<String>,
    attributes: Vec<String>,
    name: String,
) -> Vec<u8> {
    let data = format!(
        "{path}\n{}\n{}\n{attributes:?}\n{name}",
        interface.as_deref().unwrap_or_default(),
        class.as_deref().unwrap_or_default()
    );

    blake3::hash(data.as_bytes()).as_bytes().to_vec()
}

#[serde_as]
#[skip_serializing_none]
#[derive(Embed, serde::Deserialize, serde::Serialize, Debug, Clone, Default)]
pub struct CodeSnippet {
    /// If the file is in a repository, this is relative to the repo root.
    /// Otherwise, relative to the target path argument.
    pub path: String,

    /// Name of the interface/trait/etc if this is a member function
    pub interface: Option<String>,

    /// Name of the class/struct/etc if this is a member function
    pub class: Option<String>,

    pub attributes: Vec<String>,

    /// The name of this function/method/type/etc
    pub name: String,

    /// The contents of the snippet
    pub body: String,

    /// An LLM generated summary
    #[embed]
    pub summary: String,

    #[serde_as(as = "serde_with::hex::Hex")]
    pub hash: Vec<u8>,

    #[serde(skip_serializing)]
    pub rendered: String,

    pub queries: Vec<String>,
}

impl CodeSnippet {
    pub fn uuid(&self) -> Result<Uuid> {
        let CodeSnippet {
            path,
            interface,
            class,
            attributes,
            name,
            ..
        } = self;

        let hash = make_id_hash(
            path.clone(),
            interface.clone(),
            class.clone(),
            attributes.clone(),
            name.clone(),
        );

        Ok(Uuid::new_v8(hash[..16].try_into()?))
    }

    pub fn body(&self) -> Cow<String> {
        if self.rendered.is_empty() {
            Cow::Borrowed(&self.body)
        } else {
            Cow::Borrowed(&self.rendered)
        }
    }
}
