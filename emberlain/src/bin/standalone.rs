use anyhow::Result;
use emberlain::template::Templater;
use emberlain::workers::pathfinder::Pathfinder;
use emberlain::workers::progress::ProgressWorker;
use emberlain::workers::prune::PruningWorker;
use emberlain::workers::synthesize::SynthWorker;
use emberlain::{AgentFactory, LanguageMap};
use fastembed::{EmbeddingModel, ModelInfo, TextEmbedding};
use indicatif_log_bridge::LogWrapper;
use log::debug;
use qdrant_client::Qdrant;
use std::process::exit;
use std::sync::{Arc, LazyLock, Mutex};
use tokio::task::JoinSet;
use tokio::{spawn, task};
use tracing_log::LogTracer;
use tracing_subscriber::EnvFilter;

use emberlain::{
    Config, Progressor, SourceWalker, init_collection,
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

    let target_path = CONFIG.target_path.clone().unwrap();
    let target_path = std::fs::canonicalize(&target_path).unwrap_or(target_path);

    let repo_root = CONFIG.repo_root.clone().unwrap_or_else(|| {
        let repo = git2::Repository::open_ext(
            &target_path,
            git2::RepositoryOpenFlags::empty(),
            &[] as &[&std::ffi::OsStr],
        )
        .ok();

        repo.as_ref()
            .and_then(|r| r.workdir())
            .map(|p| p.to_path_buf())
            .unwrap_or(target_path.clone())
    });

    log::info!("Target dir: {target_path:?} repo root: {repo_root:?}");

    // Configure and build all workers
    let lang_specs: String = if let Some(path) = &CONFIG.lang_spec {
        std::fs::read_to_string(path)?
    } else {
        include_str!("../../etc/languages.yml").to_string()
    };

    let src_walker: SourceWalker = lang_specs.as_str().try_into()?;
    let lang_specs: Arc<LanguageMap> = Arc::new(serde_yml::from_str(&lang_specs)?);
    let templater = Templater::new(lang_specs.clone())?;

    let qdrant_client = Qdrant::from_url(CONFIG.qdrant_url.as_ref().unwrap()).build()?;
    init_collection(&qdrant_client, COLLECTION_NAME.as_str(), *EMBED_DIMS as u64).await?;

    let pathfinder = Pathfinder::builder()
        .types(src_walker.get_types()?)
        .qdrant(qdrant_client.clone())
        .collection(CONFIG.collection.clone().unwrap())
        .build();

    let mut extractor = ExtractingWorker::builder().walker(src_walker).build();

    let deduper = DedupWorker::builder()
        .templater(templater)
        .reprocess(CONFIG.reprocess.unwrap_or_default())
        .qdrant(qdrant_client.clone())
        .collection(CONFIG.collection.clone().unwrap())
        .build();

    let agent_factory = AgentFactory::new(&CONFIG);
    // agent_factory.verify().await?; // TODO: implement verification

    let summarizers = (0..CONFIG.summary_workers.unwrap_or(1))
        .map(|_| {
            let agent = agent_factory.summarizer();
            let agent = agent.unwrap().build();
            SummaryWorker::builder()
                .agent(agent)
                .reprocess(CONFIG.reprocess.unwrap_or_default())
                .dry_run(CONFIG.dry_run.unwrap_or_default())
                .build()
        })
        .collect::<Vec<_>>();

    let embed_model = TextEmbedding::try_new(
        fastembed::InitOptions::new(EMBED_MODEL.clone())
            .with_show_download_progress(true)
            .with_cache_dir(CONFIG.fastembed_cache.as_ref().unwrap().into()),
    )?;

    let embed_model = Arc::new(Mutex::new(embed_model));

    let synth_worker = SynthWorker::builder()
        .extractor(agent_factory.extractor()?.build())
        .enabled(CONFIG.synthetics.unwrap_or_default())
        .reprocess(CONFIG.reprocess.unwrap_or_default())
        .build();

    let embedder = EmbeddingWorker::builder()
        .embedding(embed_model)
        .qdrant(qdrant_client.clone())
        .collection(CONFIG.collection.clone().unwrap())
        .build();

    let progress_worker = ProgressWorker::builder().build();

    let pruner = CONFIG.pruning_cutoff()?.map(|dt| {
        PruningWorker::builder()
            .cutoff(dt)
            .qdrant(qdrant_client.clone())
            .collection(CONFIG.collection.clone().unwrap())
            .build()
    });

    // Preliminary book keeping
    let total_count = if CONFIG.progress.filter(|t| *t).is_some() {
        pathfinder
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

    let local = task::LocalSet::new();
    let (path_tx, path_rx) = flume::bounded(4);
    let (snippet_tx, snippet_rx) = flume::bounded(4);
    let (dedup_tx, dedup_rx) = flume::bounded(4);
    let (summary_tx, summary_rx) = flume::bounded(4);
    let (synth_tx, synth_rx) = flume::bounded(4);
    let (embed_tx, embed_rx) = flume::bounded(4);

    // Launch all workers
    let path_task = {
        let progressor = progressor.clone();
        let repo_root = repo_root.clone();
        let target_path = target_path.clone();

        spawn(async move {
            if let Err(err) = pathfinder
                .run(progressor, path_tx, repo_root, target_path)
                .await
            {
                log::error!("{err:?}");
                exit(1);
            }
        })
    };

    // Only tree-sitter needs to be run locally
    {
        let path_rx = path_rx.clone();
        let repo_root = repo_root.clone();
        local.spawn_local(async move {
            if let Err(err) = extractor.run(path_rx, snippet_tx, repo_root).await {
                log::error!("{err:?}");
                exit(1);
            }
        });
    }

    drop(path_rx);

    let dedup_task = spawn(async {
        if let Err(err) = deduper.run(snippet_rx, dedup_tx).await {
            log::error!("{err:?}");
            exit(1);
        }
    });

    let mut summary_tasks = JoinSet::new();

    for summarizer in summarizers {
        let dedup_rx = dedup_rx.clone();
        let summary_tx = summary_tx.clone();
        summary_tasks.spawn(async move {
            if let Err(err) = summarizer.run(dedup_rx, summary_tx).await {
                log::error!("{err:?}");
                exit(1);
            }
        });
    }

    drop(dedup_rx); // Otherwise won't automatically exit since channels still in scope
    drop(summary_tx);

    let synth_task = spawn(async move {
        if let Err(err) = synth_worker.run(summary_rx, synth_tx).await {
            log::error!("{err:?}");
            exit(1);
        }
    });

    let mut progress_channel = embed_rx;

    let embed_task = if !CONFIG.dry_run.unwrap_or(false) {
        Some(spawn(async move {
            if let Err(err) = embedder.run(synth_rx, embed_tx).await {
                log::error!("{err:?}");
                exit(1);
            }
        }))
    } else {
        progress_channel = synth_rx;
        None
    };

    let progress_task = spawn(async move {
        if let Err(err) = progress_worker.run(progress_channel).await {
            log::error!("{err:?}");
            exit(1);
        }
    });

    // This await needs to be first for anything to proceed
    local.await;
    debug!("Extraction workers done");

    debug!("Pathfinder done: {:?}", path_task.await.err());

    let dedup_errs = dedup_task.await.err();
    debug!("Dedup workers done: {dedup_errs:?}");

    let summary_errs = summary_tasks.join_all().await;
    debug!("Summary workers done: {summary_errs:?}",);

    debug!("Synthesizer worker done: {:?}", synth_task.await.err());

    if let Some(embed_task) = embed_task {
        let embed_errs = embed_task.await.err();
        debug!("Embed workers done: {embed_errs:?}");
    }

    debug!("Progress worker done: {:?}", progress_task.await.err());

    if let Some(pruner) = pruner {
        pruner.run().await?;
    }

    if let Some(bar) = progressor.as_ref() {
        bar.file_progress.abandon();
    }

    Ok(())
}
