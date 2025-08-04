use anyhow::{Result, anyhow};
use clap::Parser;
use fastembed::{ModelInfo, TextEmbedding};
use figment::{
    Figment,
    providers::{Env, Format, Serialized, Toml},
};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use indoc::indoc;
use log::info;
use log::warn;
use qdrant_client::{
    Qdrant,
    qdrant::{CreateCollectionBuilder, Distance, QueryPointsBuilder, VectorParamsBuilder},
};
use rig::embeddings::EmbeddingsBuilder;
use rig::vector_store::InsertDocuments;
use rig::{completion::Prompt, prelude::*, providers::ollama};
use rig_fastembed::Client;
use rig_fastembed::FastembedModel;
use rig_qdrant::QdrantVectorStore;
use serde::{Deserialize, Serialize};
use serde_with::skip_serializing_none;
use std::sync::{
    Arc, LazyLock,
    atomic::{AtomicUsize, Ordering},
};
use tokio::sync::mpsc::{self, Receiver};
use tokio::task;
use tracing_log::LogTracer;
use tracing_subscriber::EnvFilter;

use emberlain::{
    CodeSnippet, SourceWalker,
    parse::{cb::FileMatchArgs, process_node},
};

/// TODO: project description
#[skip_serializing_none] // This is the solution!
#[derive(Clone, Parser, Debug, Serialize, Deserialize)]
// #[command(version, about, long_about = None)]
struct Config {
    #[arg(long, action=clap::ArgAction::SetTrue)]
    dry_run: Option<bool>,

    #[arg(long, action=clap::ArgAction::SetTrue)]
    progress: Option<bool>,

    #[arg(long, action=clap::ArgAction::SetTrue)]
    dump_config: Option<bool>,

    // #[arg(long, default_value = "http://10.10.10.100:11434")] // No, masks config file
    /// Base URL of the language model server
    #[arg(long)]
    llm_base_url: Option<String>,

    /// Name of the language model for summarizing code snippets
    #[arg(long)]
    llm_model: Option<String>,

    #[arg(long)]
    fastembed_cache: Option<String>,

    #[arg(long)]
    collection: Option<String>,

    #[arg(long)]
    embed_model: Option<String>,

    target_path: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            dry_run: Default::default(),
            progress: Default::default(),
            dump_config: Default::default(),
            llm_model: Some("devstral:latest".into()),
            llm_base_url: Some("http://localhost:11434".into()),
            collection: Some("myproject".into()),
            embed_model: Default::default(),
            fastembed_cache: dirs::cache_dir().and_then(|mut d| {
                d.push("fastembed");
                d.into_os_string().into_string().ok()
            }),
            target_path: Some("./".into()),
        }
    }
}

static CONFIG: LazyLock<Config> = LazyLock::new(|| {
    Figment::new()
        .merge(Serialized::defaults(Config::default()))
        .merge(Toml::file(
            dirs::config_dir()
                .map(|p| p.join("emberlain"))
                .unwrap_or_default()
                .join("config.toml"),
        ))
        .merge(Env::prefixed("EMB_"))
        .merge(Serialized::defaults(Config::parse()))
        // TODO: select profile from env
        .extract()
        .unwrap()
});

static LLM_MODEL: LazyLock<String> = LazyLock::new(|| CONFIG.llm_model.clone().unwrap());
static LLM_BASE_URL: LazyLock<String> = LazyLock::new(|| CONFIG.llm_base_url.clone().unwrap());
static EMBED_INFO: LazyLock<ModelInfo<FastembedModel>> = LazyLock::new(|| {
    let model_name = CONFIG.embed_model.as_ref().unwrap().to_lowercase();
    let all_embeddings = TextEmbedding::list_supported_models();
    all_embeddings
        .iter()
        .find(|model| {
            model.model_code.to_lowercase().ends_with(&model_name)
                || format!("{:?}", model.model)
                    .to_lowercase()
                    .ends_with(&model_name)
        })
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "The embedding model '{}' is not valid",
                CONFIG.embed_model.as_ref().unwrap()
            )
        })
});

static EMBED_MODEL: LazyLock<FastembedModel> = LazyLock::new(|| EMBED_INFO.model.clone());
static EMBED_DIMS: LazyLock<usize> = LazyLock::new(|| EMBED_INFO.dim);
static COLLECTION_NAME: LazyLock<String> = LazyLock::new(|| CONFIG.collection.clone().unwrap());

struct Progressor {
    multi: MultiProgress,
    file_progress: ProgressBar,
}

impl Progressor {
    pub fn new() -> Self {
        let multi = MultiProgress::new();
        let file_progress = multi.add(ProgressBar::no_length());

        Self {
            multi,
            file_progress,
        }
    }
}

enum SnippetProgress {
    Snippet {
        progress: Option<ProgressBar>,
        snippet: CodeSnippet,
    },
    EndOfFile {
        progressor: Arc<Option<Progressor>>,
        progress: Option<ProgressBar>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = dotenvy::dotenv();

    if CONFIG.dump_config.filter(|b| *b).is_some() {
        let config_out = Config {
            dump_config: None,
            dry_run: None,
            ..CONFIG.clone()
        };

        println!("{}", toml::to_string(&config_out)?);
        // dbg!(&config_out);
        // dbg!(&LLM_MODEL.as_str());
        // dbg!(&*EMBED_INFO);

        return Ok(());
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // LogTracer::init()?;
    // let logger = Arc::new(LogTracer::new());
    // log::set_boxed_logger(Box::new(logger.clone()))?;
    // log::set_max_level(log::LevelFilter::Trace);
    let logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).build();
    let level = logger.filter();
    let logger = LogTracer::new();

    let (snippet_tx, mut snippet_rx) = mpsc::channel(4);
    let (summary_tx, summary_rx) = mpsc::channel(4);

    let target_dir = CONFIG.target_path.as_ref().unwrap();

    info!("Path to index: {target_dir}");
    let agent = ollama::Client::from_url(&LLM_BASE_URL)
        .agent(&LLM_MODEL)
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

    // Would like lines or bytes instead, if possible
    let total_count = if CONFIG.progress.filter(|t| *t).is_some() {
        let counter = Arc::new(AtomicUsize::new(0));
        let result = counter.clone();
        local
            .run_until(async move {
                let langspec = include_str!("../etc/languages.yml");
                let mut src_walk = SourceWalker::default();
                src_walk.load_languages(langspec)?;
                src_walk
                    .walk_directory(target_dir, &async move |_entry| {
                        counter.fetch_add(1, Ordering::Relaxed);
                    })
                    .await?;

                Ok::<_, anyhow::Error>(())
            })
            .await?;
        Some(result.load(Ordering::Relaxed))
    } else {
        None
    };

    let progressor = Arc::new(CONFIG.progress.filter(|t| *t).map(|_| Progressor::new()));
    if let Some(bars) = progressor.as_ref()
        && let Some(count) = total_count
    {
        LogWrapper::new(bars.multi.clone(), logger)
            .try_init()
            .unwrap();
        bars.file_progress.set_length(count as u64);
    }

    log::set_max_level(level);

    let pbar = progressor.clone();
    let congressor = progressor.clone();

    local.spawn_local(async move {
        let langspec = include_str!("../etc/languages.yml");
        let mut src_walk = SourceWalker::default();
        src_walk.load_languages(langspec)?;
        let pbar = pbar.clone();

        let cb = async move |entry: FileMatchArgs| {
            let progress = if let Some(bar) = pbar.as_ref() {
                bar.file_progress.inc(1);

                let byte_progress = bar
                    .multi
                    .insert_from_back(1, ProgressBar::new(entry.source.len() as u64));

                byte_progress.set_style(
                    ProgressStyle::with_template(
                        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
                    )
                    .unwrap(),
                );
                byte_progress.set_message(format!("{:?}", entry.file_path));
                Some(byte_progress)
            } else {
                dbg!(&entry.file_path);
                None
            };

            let snip_tx = snippet_tx.clone();
            let prog = progress.clone();
            let progressor = congressor.clone();

            // Why is this different from the tests?
            let root = entry.tree.root_node();

            process_node(
                root,
                entry.source,
                entry.query,
                vec![],
                &async move |node_match| {
                    let progress = prog.clone();
                    let n = node_match.query_match;
                    let p = entry.file_path;
                    let q = entry.query;
                    let src = entry.source;
                    info!("^_- Match {n:?} at {p:?}");
                    let mut class: Option<String> = None;
                    let mut ident: Option<String> = None;
                    let mut kind: Option<String> = None;
                    let mut body: Option<String> = None;

                    // Maybe match destructuring should be part of SourceWalker
                    for cap in &n.captures {
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
                            ["name", "reference", _] => {
                                if let Ok(n) = cap.node.utf8_text(src) {
                                    class = Some(n.to_string());
                                }
                            }
                            _ => {
                                warn!("Don't know what to do with this capture: {cap_name}")
                            }
                        }
                    }

                    info!("o.O Match results kind: {kind:?} identier: {ident:?}");
                    if let Some(body) = &body {
                        let snippet = CodeSnippet {
                            path: p.display().to_string(),
                            class,
                            name: ident.clone().unwrap_or("???".to_string()),
                            body: body.clone(),
                            summary: "".to_string(),
                        };

                        let msg = SnippetProgress::Snippet {
                            progress: progress.clone(),
                            snippet,
                        };

                        snip_tx.send(msg).await.unwrap();
                    }
                },
            )
            .await;

            snippet_tx
                .send(SnippetProgress::EndOfFile {
                    progressor: progressor.clone(),
                    progress: progress.clone(),
                })
                .await
                .unwrap();
        };

        src_walk.walk_directory(target_dir, &cb).await?;
        info!("Done walking");

        Ok::<_, anyhow::Error>(())
    });

    local.spawn_local(async move {
        // TODO: check and skip existing snippets
        // Not worth batching when using ollama with such weak hardware, but later...
        while let Some(msg) = snippet_rx.recv().await {
            match msg {
                SnippetProgress::Snippet {
                    progress, snippet, ..
                } => {
                    // Skip one-liners, aliases, forward declarations, etc.
                    if !snippet.body.contains("\n") {
                        continue;
                    }

                    let count = snippet.body.len();

                    // TODO: strip out comments since we want to rely on LLM to interpret code rather than
                    // regurgitating out-of-date or deceptive descriptions
                    let body = if let Some(self_type) = &snippet.class {
                        // TODO: universal commenting or wrap in markdown
                        format!("/// self: {self_type}\n{}", &snippet.body)
                    } else {
                        snippet.body.clone()
                    };

                    let options = textwrap::Options::new(100)
                        .initial_indent(">>> ")
                        .subsequent_indent("... ");
                    info!("{}", textwrap::fill(&body, &options));

                    if !CONFIG.dry_run.unwrap_or(false) {
                        match agent.prompt(body).await {
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

                    // Would prefer using file position since there could be large comments at the
                    // top level, but parser can jump around quite a bit depending on the traversal.
                    progress.as_ref().inspect(|p| p.inc(count as u64));
                }
                SnippetProgress::EndOfFile {
                    progressor,
                    progress,
                } => {
                    if let Some(bar) = progress.as_ref()
                        && let Some(Progressor { multi, .. }) = progressor.as_ref()
                    {
                        // Emulate detaching a finished bar from the multi by creating a dummy
                        multi.remove(bar);
                        multi.suspend(|| {
                            let tombstone = bar
                                .length()
                                .map(ProgressBar::new)
                                .unwrap_or_else(ProgressBar::no_length)
                                .with_prefix(bar.prefix())
                                .with_style(bar.style())
                                .with_elapsed(bar.elapsed());

                            tombstone.finish_with_message(bar.message());
                        });
                    }
                }
            }
        }

        info!("No more snippets to summarize");
    });

    if !CONFIG.dry_run.unwrap_or(false) {
        local.spawn_local(async move {
            embedsert(summary_rx).await.unwrap();
        });
    } else {
        drop(summary_rx);
    }

    local.await;

    if let Some(bar) = progressor.as_ref() {
        bar.file_progress.abandon();
    }

    Ok(())
}

async fn embedsert(mut summary_rx: Receiver<CodeSnippet>) -> Result<()> {
    // Initialize the Fastembed client
    let fastembed_client = Client::new();

    let embedding_model = fastembed_client.embedding_model(&EMBED_MODEL);

    let client = Qdrant::from_url("http://localhost:6334").build()?;
    if !client.collection_exists(COLLECTION_NAME.as_str()).await? {
        client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME.as_str()).vectors_config(
                    VectorParamsBuilder::new(*EMBED_DIMS as u64, Distance::Cosine),
                ),
            )
            .await?;
    }

    let query_params = QueryPointsBuilder::new(COLLECTION_NAME.as_str()).with_payload(true);
    let vector_store =
        QdrantVectorStore::new(client, embedding_model.clone(), query_params.build());

    // TODO: continue after error in loop
    while let Some(msg) = summary_rx.recv().await {
        // println!("Summarized: {msg:?}");
        let options = textwrap::Options::new(100)
            .initial_indent(">.< ")
            .subsequent_indent("-.- ");

        info!("{}", textwrap::fill(&msg.summary, &options));

        // TODO: robust error handling
        let documents = EmbeddingsBuilder::new(embedding_model.clone())
            .document(msg)?
            .build()
            .await?;

        // TODO: how to set the ID via the Rig wrapper?
        vector_store
            .insert_documents(documents)
            .await
            .map_err(|err| anyhow!("Couldn't insert documents: {err}"))?;
    }

    Ok(())
}
