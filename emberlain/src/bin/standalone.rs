use anyhow::Result;
use indicatif_log_bridge::LogWrapper;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task;
use tracing_log::LogTracer;
use tracing_subscriber::EnvFilter;

use emberlain::{
    CONFIG, Config, Progressor,
    workers::{
        embed::embedding_worker,
        extract::{counting_worker, extracting_worker},
        summarize::summary_worker,
    },
};

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

    let logger =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).build();
    let level = logger.filter();
    let logger = LogTracer::new();

    let progressor = Arc::new(
        CONFIG
            .progress
            .filter(|t| *t)
            .map(|_| Progressor::default()),
    );
    if let Some(bars) = progressor.as_ref() {
        LogWrapper::new(bars.multi.clone(), logger)
            .try_init()
            .unwrap();
    }

    log::set_max_level(level);

    let local = task::LocalSet::new();

    let total_count = if CONFIG.progress.filter(|t| *t).is_some() {
        local
            .run_until(counting_worker(CONFIG.target_path.as_ref().unwrap()))
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

    let (snippet_tx, snippet_rx) = mpsc::channel(4);
    let (summary_tx, summary_rx) = mpsc::channel(4);

    local.spawn_local(extracting_worker(
        progressor.clone(),
        snippet_tx,
        CONFIG.target_path.as_ref().unwrap(),
    ));

    local.spawn_local(summary_worker(snippet_rx, summary_tx));

    if !CONFIG.dry_run.unwrap_or(false) {
        local.spawn_local(async move {
            embedding_worker(summary_rx).await.unwrap();
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
