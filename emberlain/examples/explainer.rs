use anyhow::{Result, anyhow};
use indoc::indoc;
use qdrant_client::{
    Qdrant,
    qdrant::{CreateCollectionBuilder, Distance, QueryPointsBuilder, VectorParamsBuilder},
};
use rig::embeddings::EmbeddingsBuilder;
use rig::{Embed, vector_store::InsertDocuments};
use rig::{completion::Prompt, prelude::*, providers::ollama};
use rig_fastembed::Client;
use rig_fastembed::FastembedModel;
use rig_qdrant::QdrantVectorStore;
use tokio::sync::mpsc::{self, Receiver};
use tokio::task;

use emberlain::SourceWalker;
use tracing::warn;

const LLM_MODEL: &str = "devstral:latest";
const LLM_BASE_URL: &str = "http://10.10.10.100:11434";
const COLLECTION_NAME: &str = "chipsndip";
const EMBED_MODEL: FastembedModel = FastembedModel::MxbaiEmbedLargeV1Q;
const EMBEDDING_DIMS: u64 = 1024;

#[derive(Embed, serde::Deserialize, serde::Serialize, Debug)]
struct CodeSnippet {
    path: String,
    name: String,
    body: String,

    #[embed]
    summary: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().init();
    let (snippet_tx, mut snippet_rx) = mpsc::channel(4);
    let (summary_tx, summary_rx) = mpsc::channel(4);

    let args = std::env::args().collect::<Vec<_>>();
    let target_dir = if args.len() > 1 {
        args[1].clone()
    } else {
        "./".to_string()
    };

    tracing::info!("Path to index: {target_dir}");

    let agent = ollama::Client::from_url(LLM_BASE_URL)
        .agent(LLM_MODEL)
        .max_tokens(1024)
        // TODO: Really need to work on taming the LLM's verbiage
        .preamble(indoc! {r##"
            You are a helpful software engineer mentoring a new team-mate.
            Without preamble or introduction, summarize provided code snippets, in a few sentences.
            Explain what it does and how it works in general terms without referring to specific values.
            If it uses higher level concepts and design patterns like observers, pipelines, etc. make note of that.
            Be sure to mention key types and functions used, if applicable.
            But do not dwell on basic concepts or things that are obvious like what it means for something to be public.
            Keep your explanation in paragraph format, using complete sentences.
        ""##})
        .build();

    let local = task::LocalSet::new();

    // Requires local since tree-sitter handles cannot be used multi-threaded
    local.spawn_local(async move {
        let langspec = include_str!("../etc/languages.yml");
        let mut src_walk = SourceWalker::default();
        src_walk.load_languages(langspec)?;
        src_walk
            .aprocess_directory(target_dir, async move |p, q, n, src| {
                println!("^_- Match {n:?} at {p:?}");
                let mut ident: Option<String> = None;
                let mut kind: Option<String> = None;
                let mut body: Option<String> = None;

                // Maybe match destructuring should be part of SourceWalker
                for cap in n.captures {
                    let index = cap.index as usize;
                    if q.capture_names().len() <= index {
                        continue;
                    }

                    let cap_name = q.capture_names()[index];
                    let parts: Vec<&str> = cap_name.split(".").collect();
                    match parts.as_slice() {
                        ["definition", k] => {
                            kind = Some(k.to_string());
                            if let Ok(n) = cap.node.utf8_text(src) {
                                body = Some(n.to_string());
                            }
                        }
                        ["name", "definition", _] => {
                            if let Ok(n) = cap.node.utf8_text(src) {
                                ident = Some(n.to_string());
                            }
                        }
                        _ => tracing::warn!("Don't know what to do with this capture: {cap_name}"),
                    }
                }

                println!("o.O Match results kind: {kind:?} identier: {ident:?}");
                if let Some(body) = &body {
                    let snippet = CodeSnippet {
                        path: p.display().to_string(),
                        name: ident.clone().unwrap_or("???".to_string()),
                        body: body.clone(),
                        summary: "".to_string(),
                    };

                    snippet_tx.send(snippet).await.unwrap();
                }
            })
            .await?;

        Ok(()) as anyhow::Result<()>
    });

    local.spawn_local(async move {
        // Not worth batching when using ollama with such weak hardware, but later...
        while let Some(snippet) = snippet_rx.recv().await {
            // Skip one-liners, aliases, forward declarations, etc.
            if !snippet.body.contains("\n") {
                continue;
            }

            // TODO: error handling
            match agent.prompt(&snippet.body).await {
                Ok(resp) => {
                    let snippet = CodeSnippet {
                        summary: resp,
                        ..snippet
                    };
                    summary_tx.send(snippet).await.unwrap();
                }
                Err(err) => warn!("Could not summarize snippet: {err:?}"),
            }
        }
    });

    local.spawn_local(async move {
        embedsert(summary_rx).await.unwrap();
    });

    local.await;

    Ok(())
}

async fn embedsert(mut summary_rx: Receiver<CodeSnippet>) -> Result<()> {
    // Initialize the Fastembed client
    let fastembed_client = Client::new();

    let embedding_model = fastembed_client.embedding_model(&EMBED_MODEL);

    let client = Qdrant::from_url("http://localhost:6334").build()?;
    if !client.collection_exists(COLLECTION_NAME).await? {
        client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME)
                    .vectors_config(VectorParamsBuilder::new(EMBEDDING_DIMS, Distance::Cosine)),
            )
            .await?;
    }

    let query_params = QueryPointsBuilder::new(COLLECTION_NAME).with_payload(true);
    let vector_store =
        QdrantVectorStore::new(client, embedding_model.clone(), query_params.build());

    // TODO: continue after error in loop
    while let Some(msg) = summary_rx.recv().await {
        let options = textwrap::Options::new(100)
            .initial_indent(">>> ")
            .subsequent_indent("... ");
        println!("{}", textwrap::fill(&msg.body, &options));

        // println!("Summarized: {msg:?}");
        let options = textwrap::Options::new(100)
            .initial_indent(">.< ")
            .subsequent_indent("-.- ");

        println!("{}", textwrap::fill(&msg.summary, &options));

        // TODO: robust error handling
        let documents = EmbeddingsBuilder::new(embedding_model.clone())
            .document(msg)?
            .build()
            .await?;

        vector_store
            .insert_documents(documents)
            .await
            .map_err(|err| anyhow!("Couldn't insert documents: {err}"))?;
    }

    Ok(())
}
