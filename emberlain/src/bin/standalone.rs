use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, ModelInfo, TextEmbedding};
use indicatif_log_bridge::LogWrapper;
use itertools::Itertools;
use log::debug;
use qdrant_client::{
    Qdrant,
    qdrant::{CreateCollectionBuilder, Distance, VectorParamsBuilder},
};
use std::sync::{Arc, LazyLock};
use tokio::task::JoinSet;
use tokio::{spawn, task};
use tracing_log::LogTracer;
use tracing_subscriber::EnvFilter;

use emberlain::{
    Config, Progressor, SnippetProgress, SourceWalker, get_agent_factory,
    workers::{
        dedup::DedupWorker, embed::EmbeddingWorker, extract::ExtractingWorker,
        summarize::SummaryWorker,
    },
};

pub static CONFIG: LazyLock<Config> = LazyLock::new(|| Config::load().unwrap());

pub static EMBED_INFO: LazyLock<ModelInfo<EmbeddingModel>> = LazyLock::new(|| {
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

pub static EMBED_MODEL: LazyLock<EmbeddingModel> = LazyLock::new(|| EMBED_INFO.model.clone());
pub static EMBED_DIMS: LazyLock<usize> = LazyLock::new(|| EMBED_INFO.dim);
pub static COLLECTION_NAME: LazyLock<String> = LazyLock::new(|| CONFIG.collection.clone().unwrap());

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

        return Ok(());
    }

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // TODO: clean this up
    let progressor = Arc::new(
        CONFIG
            .progress
            .filter(|t| *t)
            .map(|_| Progressor::default()),
    );
    if let Some(bars) = progressor.as_ref() {
        let logger =
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
                .build();
        let level = logger.filter();
        let logger = LogTracer::new();

        LogWrapper::new(bars.multi.clone(), logger)
            .try_init()
            .unwrap();
        log::set_max_level(level);
    } else {
        LogTracer::init()?;
    }

    // Configure and build all workers
    let src_walker: SourceWalker = if let Some(path) = &CONFIG.lang_spec {
        std::fs::read_to_string(path)?.as_str().try_into()?
    } else {
        include_str!("../../etc/languages.yml").try_into()?
    };

    let mut extractor = ExtractingWorker::builder().walker(src_walker).build();
    reqwest::get(CONFIG.llm_base_url.as_ref().unwrap())
        .await
        .context("Unable to connect to LLM provider")?
        .text()
        .await?;

    let deduper = DedupWorker::builder()
        .reprocess(CONFIG.reprocess.unwrap_or_default())
        .build();

    let agent_factory = get_agent_factory(&CONFIG)?;
    agent_factory.check().await?;

    let summarizers = (0..CONFIG.summary_workers.unwrap_or(1))
        .map(|_| {
            SummaryWorker::builder()
                .agent(agent_factory.build())
                .dry_run(CONFIG.dry_run.unwrap_or_default())
                .build()
        })
        .collect::<Vec<_>>();

    let embed_model = Arc::new(TextEmbedding::try_new(
        fastembed::InitOptions::new(EMBED_MODEL.clone())
            .with_show_download_progress(true)
            .with_cache_dir(CONFIG.fastembed_cache.as_ref().unwrap().into()),
    )?);

    let qdrant_client = Qdrant::from_url(CONFIG.qdrant_url.as_ref().unwrap()).build()?;
    if !qdrant_client
        .collection_exists(COLLECTION_NAME.as_str())
        .await?
    {
        qdrant_client
            .create_collection(
                CreateCollectionBuilder::new(COLLECTION_NAME.as_str()).vectors_config(
                    VectorParamsBuilder::new(*EMBED_DIMS as u64, Distance::Cosine),
                ),
            )
            .await?;
        // TODO: manage indices
    }

    let embedder = EmbeddingWorker::builder()
        .embedding(embed_model.clone())
        .qdrant(qdrant_client.clone())
        .collection(CONFIG.collection.clone().unwrap())
        .build();

    // Preliminary book keeping
    let total_count = if CONFIG.progress.filter(|t| *t).is_some() {
        extractor
            .count_files(CONFIG.target_path.as_ref().unwrap())
            .await
            .ok()
    } else {
        None
    };

    if let Some(bars) = progressor.as_ref()
        && let Some(count) = total_count
    {
        bars.file_progress.set_length(count as u64);
    }

    // hmm, is capacity per receiver, sender or total?
    let (snippet_tx, snippet_rx) = flume::bounded::<SnippetProgress>(4);
    let (dedup_tx, dedup_rx) = flume::bounded::<SnippetProgress>(4);
    let (summary_tx, summary_rx) = flume::bounded(4);

    // Launch all workers

    // Only tree-sitter needs to be run locally
    let local = task::LocalSet::new();
    {
        let progressor = progressor.clone();
        local.spawn_local(async move {
            extractor
                .run(progressor, snippet_tx, CONFIG.target_path.as_ref().unwrap())
                .await
        });
    }

    let dedup_task = spawn(deduper.run(snippet_rx, dedup_tx));

    let mut summary_tasks = JoinSet::new();

    for summarizer in summarizers {
        let dedup_rx = dedup_rx.clone();
        let summary_tx = summary_tx.clone();
        summary_tasks.spawn(async move { summarizer.run(dedup_rx, summary_tx).await });
    }

    drop(dedup_rx); // Otherwise won't automatically exit since channels still in scope
    drop(summary_tx);

    let embed_task = if !CONFIG.dry_run.unwrap_or(false) {
        Some(spawn(async move {
            embedder.run(summary_rx).await.unwrap();
        }))
    } else {
        drop(summary_rx);
        None
    };

    local.await;
    debug!("Extraction workers done");

    let dedup_errs = dedup_task.await.err();
    debug!("Dedup workers done: {dedup_errs:?}");

    let summary_errs = summary_tasks
        .join_all()
        .await
        .into_iter()
        .map(|r| r.err())
        .collect_vec();
    debug!("Summary workers done: {summary_errs:?}",);

    if let Some(embed_task) = embed_task {
        let embed_errs = embed_task.await.err();
        debug!("Embed workers done: {embed_errs:?}");
    }

    if let Some(bar) = progressor.as_ref() {
        bar.file_progress.abandon();
    }

    Ok(())
}
