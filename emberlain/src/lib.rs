use anyhow::{Result, anyhow};
use cached_path::cached_path;
use ignore::Walk;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;
use tokio::fs::File;
use tokio::io::AsyncReadExt;
use tree_sitter::{Language, Parser, Query, QueryCursor, WasmStore, wasmtime::Engine};
use tree_sitter::{QueryMatch, StreamingIterator};

use tokio::task;

#[derive(Serialize, Deserialize, Debug)]
pub struct LanguageSpec {
    pub grammar: String,
    pub queries: BTreeMap<String, String>,
    pub extensions: Vec<String>,
    pub enabled: Option<bool>,
}

pub struct CodeSnipper {
    pub name: String,
    pub blob: Language,
    pub parser: Parser,
    pub query: Query,
}

#[derive(Default)]
pub struct SourceWalker {
    pub engine: Engine,
    pub languages: BTreeMap<String, LanguageSpec>,
    pub ext_to_lang: BTreeMap<String, String>,
    pub snippers: BTreeMap<String, CodeSnipper>,
}

impl SourceWalker {
    pub fn new(engine: Engine) -> Self {
        Self {
            engine,
            languages: BTreeMap::new(),
            ext_to_lang: BTreeMap::new(),
            snippers: BTreeMap::new(),
        }
    }

    pub fn load_languages(&mut self, langspec: &str) -> anyhow::Result<()> {
        self.languages = serde_yml::from_str(langspec)?;
        self.ext_to_lang = self
            .languages
            .iter()
            .flat_map(|(k, v)| v.extensions.iter().map(|x| (x.clone(), k.clone())))
            .collect();
        Ok(())
    }

    async fn make_processor(
        engine: &Engine,
        lang_name: String,
        lang_spec: &LanguageSpec,
    ) -> Result<CodeSnipper> {
        let grammar_url = lang_spec.grammar.clone();
        let grammar_path = task::spawn_blocking(move || cached_path(&grammar_url)).await??;

        let mut grammar_file = File::open(grammar_path).await?;
        let mut grammar_buf = Vec::new();
        grammar_file.read_to_end(&mut grammar_buf).await?;

        let mut store = WasmStore::new(engine)?;
        let language = store.load_language(&lang_name, &grammar_buf).unwrap();

        let mut parser = Parser::new();
        parser.set_wasm_store(store)?;
        parser.set_language(&language)?;

        let query = lang_spec.queries.values().join("\n");
        let query = Query::new(&language, &query)?;

        Ok(CodeSnipper {
            name: lang_name.clone(),
            blob: language,
            parser,
            query,
        })
    }

    pub async fn snipper_for(&mut self, file_ext: &str) -> Result<&mut CodeSnipper> {
        let lang_name = self
            .ext_to_lang
            .get(file_ext)
            .ok_or(anyhow!("Extension '{file_ext}' is not supported"))?;

        if !self.snippers.contains_key(lang_name) {
            let lang_spec = self
                .languages
                .get(lang_name)
                .ok_or(anyhow!("Language '{lang_name}' is not supported"))?;

            self.snippers.insert(
                lang_name.clone(),
                Self::make_processor(&self.engine, lang_name.clone(), lang_spec).await?,
            );
        }

        self.snippers
            .get_mut(lang_name)
            .ok_or(anyhow!("Could not retrieve processor for {file_ext}"))
    }

    pub async fn process_matches<P: AsRef<Path>>(
        &mut self,
        path: P,
        cb: impl AsyncFn(&QueryMatch, &[u8]),
    ) -> Result<()> {
        for item in Walk::new(path) {
            match item {
                Ok(entry) if entry.path().is_file() => {
                    tracing::debug!("{entry:?} ext: {:?}", entry.path().extension());

                    if let Some(file_ext) = entry.path().extension().and_then(|x| x.to_str()) {
                        let snipper = self.snipper_for(file_ext).await;

                        if let Ok(snipper) = snipper {
                            let parser = &mut snipper.parser;
                            let query = &snipper.query;

                            let mut source_code: Vec<u8> = Vec::new();
                            let mut fh = File::open(entry.path()).await?;
                            fh.read_to_end(&mut source_code).await?;

                            let tree = parser
                                .parse(&source_code, None)
                                .ok_or(anyhow!("Could not parse"))?;

                            let mut qc = QueryCursor::new();
                            tracing::info!("Query {query:?}, {}", qc.match_limit());

                            let mut ms =
                                qc.matches(query, tree.root_node(), source_code.as_slice());

                            while let Some(n) = ms.next() {
                                cb(n, &source_code).await;
                            }
                        }
                    }
                }
                Err(err) => tracing::warn!("Error: {err}"),
                it => tracing::debug!("Skipping {it:?}"),
            }
        }
        Ok(())
    }
}
