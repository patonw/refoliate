use std::fs::File;
use std::io::Read;

use cached_path::cached_path;
use tokio::task;
use tree_sitter::{Language, Parser, WasmStore, wasmtime::Engine};

pub const TREE_SITTER_RUST: &str = "https://github.com/tree-sitter/tree-sitter-rust/releases/download/v0.24.0/tree-sitter-rust.wasm";

pub fn load_language(lang_name: &str, grammar_url: &str) -> anyhow::Result<(Language, Parser)> {
    let engine = Engine::default();
    let grammar_path = cached_path(grammar_url)?;

    let mut grammar_file = File::open(grammar_path)?;
    let mut grammar_buf = Vec::new();
    grammar_file.read_to_end(&mut grammar_buf)?;

    let mut store = WasmStore::new(&engine)?;
    let language = store.load_language(lang_name, &grammar_buf).unwrap();

    let mut parser = Parser::new();
    parser.set_wasm_store(store)?;
    parser.set_language(&language)?;
    Ok((language, parser))
}

pub async fn aload_language(
    lang_name: &str,
    grammar_url: &str,
) -> anyhow::Result<(Language, Parser)> {
    let lang_name = lang_name.to_string();
    let grammar_url = grammar_url.to_string();
    task::spawn_blocking(move || load_language(&lang_name, &grammar_url)).await?
}
