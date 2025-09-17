use anyhow::Context as _;
use cached::proc_macro::cached;
use fastembed::{RerankInitOptions, RerankerModel, TextEmbedding, TextRerank};
use itertools::Itertools;
use qdrant_client::{
    Qdrant,
    qdrant::{
        Condition, Filter, QueryPointsBuilder, vectors_config::Config as VecConfig,
        with_payload_selector::SelectorOptions,
    },
};
use rmcp::{
    Json, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    schemars::JsonSchema,
    serde::{Deserialize, Serialize},
    serde_json::{self, Value, json},
    tool, tool_handler, tool_router,
    transport::stdio,
};
use serde_with::skip_serializing_none;
use std::{
    sync::{Arc, Mutex},
    time::Duration,
};
use typed_builder::TypedBuilder;

use crate::config::{Config, get_embed_info};

mod config;

#[skip_serializing_none]
#[derive(Serialize, Deserialize, JsonSchema)]
struct SearchRequest {
    /// Text of the query
    text: String,

    /// Number of results to return (default: 5)
    limit: Option<u64>,

    /// Additional number of points to fetch. If non zero, will rerank results. (default: 5)
    overfetch: Option<u64>,

    /// Exclude points that have these attribute, delimited by ";" (default: "test; tokio::test")
    exclude: Option<String>,

    /// Payload keys to return, delimited by ";" (default: "interface; class; name; path; summary")
    fields: Option<String>,
}

#[derive(Serialize, Deserialize, JsonSchema)]
struct SearchResponse {
    data: Vec<Value>,
}

#[derive(TypedBuilder)]
pub struct QdrantTool {
    #[builder(default=QdrantTool::tool_router())]
    tool_router: ToolRouter<QdrantTool>,

    embedder: Arc<Mutex<TextEmbedding>>,

    reranker: Arc<Mutex<TextRerank>>,

    client: Qdrant,

    collection: String,
}

#[cached(
    convert = r##"{ format!("{collection}") }"##,
    key = "String",
    time = 10,
    result = true
)]
async fn get_vectors_config(client: &Qdrant, collection: String) -> anyhow::Result<VecConfig> {
    let meta = client.collection_info(collection).await?;
    let vectors_config: VecConfig = meta
        .result
        .context("No result")?
        .config
        .context("No config")?
        .params
        .context("No params")?
        .vectors_config
        .context("No vectors config")?
        .config
        .context("No config")?;

    Ok(vectors_config)
}

#[tool_router]
impl QdrantTool {
    #[tool]
    /// Query repository for code snippets by semantic similarity
    async fn search_repo(
        &self,
        params: Parameters<SearchRequest>,
    ) -> Result<Json<SearchResponse>, String> {
        let Parameters(SearchRequest {
            text,
            limit,
            overfetch,
            exclude,
            fields,
        }) = params;

        let vec_config = get_vectors_config(&self.client, self.collection.clone())
            .await
            .map_err(|e| e.to_string())?;
        let mut embeds = {
            let mut embedder = self.embedder.lock().map_err(|e| e.to_string())?;
            embedder
                .embed(vec![text.clone()], None)
                .map_err(|e| e.to_string())?
        };

        let embedding = embeds.remove(0);

        let num_results = limit.unwrap_or(5);
        let num_fetch = num_results + overfetch.unwrap_or(5);

        let excluded_attrs = exclude
            .as_deref()
            .unwrap_or("test; tokio::test")
            .split(";")
            .map(|s| s.trim().to_string())
            .collect_vec();

        let point_filter = Filter {
            must: vec![Condition::is_empty("__removed")],
            must_not: vec![Condition::matches("attributes", excluded_attrs)],
            ..Default::default()
        };

        let field_selector = fields
            .as_deref()
            .unwrap_or("interface; class; name; path; summary")
            .split(";")
            .map(|s| s.trim().to_string())
            .collect_vec();

        let query = QueryPointsBuilder::new(self.collection.as_str())
            .query(embedding.clone())
            .with_payload(SelectorOptions::Include(field_selector.into()))
            .filter(point_filter)
            .limit(num_fetch);

        let query = if let VecConfig::ParamsMap(_params) = vec_config {
            // TODO: pull alias from config
            // TODO: Check params has key
            query.using("aliases")
        } else {
            query
        };

        let resp = self.client.query(query).await.unwrap();

        let texts = resp
            .result
            .iter()
            .map(|p| {
                p.payload
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&"".to_string())
                    .to_string()
            })
            .collect_vec();

        let mut data = resp
            .result
            .iter()
            .filter_map(|point| {
                serde_json::to_value(json!({"payload": &point.payload, "score": point.score})).ok()
            })
            .collect_vec();

        let data = if num_fetch > num_results {
            let mut reranker = self.reranker.lock().map_err(|e| e.to_string())?;
            let results = reranker
                .rerank(text, texts, true, None)
                .map_err(|e| e.to_string())?;

            results
                .iter()
                .map(|r| data[r.index].take())
                .take(num_results as usize)
                .collect_vec()
        } else {
            data
        };

        Ok(Json(SearchResponse { data }))
    }
}

// Implement the server handler
#[tool_handler]
impl rmcp::ServerHandler for QdrantTool {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            instructions: Some(format!(
                "Semantic database of code fragments for the **{}** project",
                &self.collection
            )),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

// Run the server
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = Config::load()?;

    if config.dump_config.unwrap_or_default() {
        let config_out = Config {
            dump_config: None,
            ..config.clone()
        };

        println!("{}", toml::to_string(&config_out)?);

        return Ok(());
    }

    let embed_info = get_embed_info(&config).unwrap();
    let embedder = TextEmbedding::try_new(
        fastembed::InitOptions::new(embed_info.model.clone())
            .with_show_download_progress(true)
            .with_cache_dir(config.fastembed_cache.as_ref().unwrap().into()),
    )?;

    let reranker = TextRerank::try_new(
        RerankInitOptions::new(RerankerModel::JINARerankerV1TurboEn)
            .with_show_download_progress(true)
            .with_cache_dir(config.fastembed_cache.as_ref().unwrap().into()),
    )?;

    let client = Qdrant::from_url(config.qdrant_url.as_ref().unwrap()).build()?;

    let handler = QdrantTool::builder()
        .embedder(Arc::new(Mutex::new(embedder)))
        .reranker(Arc::new(Mutex::new(reranker)))
        .client(client)
        .collection(config.collection.clone().unwrap())
        .build();

    // Create and run the server with STDIO transport
    let service = handler.serve(stdio()).await.inspect_err(|e| {
        println!("Error starting server: {e}");
    })?;
    service.waiting().await?;

    Ok(())
}
