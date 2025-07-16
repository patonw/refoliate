use anyhow::Result;
use cached::proc_macro::cached;
use egui::emath::Numeric;
use fastembed::{EmbeddingModel, TextEmbedding};
use itertools::{Itertools, izip};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::vectors_output::VectorsOptions;
use qdrant_client::qdrant::{
    GetPointsBuilder, QueryPointsBuilder, ScrollPointsBuilder, vectors_config,
};
use std::collections::{BTreeMap, HashMap};
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{LazyLock, Mutex};

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::PyDataFrame;
use tokio::runtime::Runtime;

use eframe::egui;
use egui::{
    Align, CollapsingHeader, Color32, Frame, Layout, RichText, ScrollArea, Sense, UiBuilder,
};
use egui_plot::{MarkerShape, Plot, PlotResponse, Points};

use embasee::{optzip, pydict, pyimport};

static UMAP: LazyLock<Py<PyAny>> = LazyLock::new(|| pyimport!("umap", "UMAP").unwrap());

static UMAP_NEIGHBORS: LazyLock<u64> = LazyLock::new(|| {
    env::var("UMAP_NEIGHBORS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(3)
});

const PALETTE: colorous::Gradient = colorous::ORANGE_RED;
static VECSTORE_URL: LazyLock<String> =
    LazyLock::new(|| env::var("VECSTORE_URL").unwrap_or("http://localhost:6334".to_string()));

fn main() -> anyhow::Result<()> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).

    // Necessary to keep an instance in memory to prevent SIGSEGV, even if it's not used.
    // Guessing UMAP implementation is using ref counts to keep C data in memory.
    let _umap = Python::with_gil(|py| {
        let umap = UMAP.bind(py).call(
            (),
            Some(&pydict! { py;
                // "n_neighbors" => 5
            }),
        )?;
        Ok::<_, PyErr>(umap.unbind())
    })?;

    // TODO: a warm-up fitting in a background thread

    let native_options = eframe::NativeOptions::default();

    eframe::run_native(
        "My egui App",
        native_options,
        Box::new(|cc| Ok(Box::new(MyEguiApp::new(cc)))),
    )
    .unwrap();
    println!("ByBye");

    Ok(())
}

// #[derive(Default)]
// struct Reduction {
//     umap: Option<Py<PyAny>>,
// }

#[derive(Debug, Clone)]
struct SemanticQuery {
    text: String,
    embed_model: EmbeddingModel,
    matched_ids: Arc<BTreeMap<String, f32>>,
    query_point: Option<(f64, f64)>,
}

impl Default for SemanticQuery {
    fn default() -> Self {
        Self {
            text: Default::default(),
            embed_model: EmbeddingModel::AllMiniLML6V2Q,
            matched_ids: Default::default(),
            query_point: Default::default(),
        }
    }
}

#[derive(Default, Debug, Clone)]
struct AppState {
    umap_df: DataFrame,
    hash_to_uuid: HashMap<egui::Id, String>,
    hover_point: Option<String>,
    select_point: Option<String>,
    point_details: BTreeMap<String, String>,
    semantic: SemanticQuery,
    available_collections: Arc<Vec<String>>,
    collection_name: Option<String>,
    embed_dims: usize,
}

impl AppState {
    pub fn new() -> Self {
        let umap_df = df! {
            "uuid" => vec![""; 0],
            "umap0" => vec![0f32; 0],
            "umap1" => vec![0f32; 0],
        }
        .unwrap();

        Self {
            umap_df,
            embed_dims: 384,
            ..Default::default()
        }
    }
}

struct MyEguiApp {
    // running: AtomicBool,
    rt: Runtime,
    qdclient: Arc<Qdrant>,
    app_state: Arc<Mutex<AppState>>,
    task_count: Arc<AtomicU16>,
    // TODO: refactor into Reduction
    umap: Arc<Mutex<Option<Py<PyAny>>>>,
    // reduction: Arc<Mutex<Reduction>>,
}

impl MyEguiApp {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();

        let qdclient = Arc::new(Qdrant::from_url(VECSTORE_URL.as_str()).build().unwrap());

        let mut this = Self {
            rt,
            qdclient,
            app_state: Arc::new(Mutex::new(AppState::new())),
            task_count: Default::default(),
            umap: Arc::new(Mutex::new(None)),
            // reduction: Arc::new(Mutex::new(Default::default())),
        };

        this.refresh_points();
        this.refresh_collections();

        this
    }

    fn refresh_collections(&mut self) {
        let app_state = self.app_state.clone();
        let qdclient = self.qdclient.clone();

        self.rt.handle().spawn(async move {
            if let Ok(collections) = qdclient.list_collections().await {
                let names = Arc::new(
                    collections
                        .collections
                        .into_iter()
                        .map(|c| c.name)
                        .collect::<Vec<_>>(),
                );

                log::info!("List collections: {names:?}");

                if let Ok(mut app_state) = app_state.lock() {
                    app_state.available_collections = names.clone();
                }

                refresh_collection_info(app_state, qdclient).await;
            }
        });
    }

    // TODO: only get vectors for new IDs
    fn refresh_points(&mut self) {
        let rt = self.rt.handle().to_owned();
        let app_lock = self.app_state.clone();
        let qdclient = self.qdclient.clone();
        let task_count = self.task_count.clone();
        let umap_lock = self.umap.clone();

        let collection_name = if let Ok(app_state) = self.app_state.lock() {
            app_state.collection_name.clone()
        } else {
            None
        };

        if collection_name.is_some() {
            log::info!("Refreshing");
        } else {
            log::info!("No collection selected. Skipping");
            return;
        }

        let collection_name = collection_name.unwrap();
        task_count.fetch_add(1, Ordering::Relaxed);

        self.rt.handle().spawn(async move {
            let resp = qdclient
                .scroll(
                    ScrollPointsBuilder::new(collection_name.as_str())
                        .limit(1000)
                        .with_payload(true)
                        .with_vectors(true),
                )
                .await;

            let num_points = resp.as_ref().unwrap().result.len();
            log::info!("Found {num_points} results");

            if num_points > 0 {
                let point_vecs: Vec<_> = resp
                    .as_ref()
                    .unwrap()
                    .result
                    .iter()
                    .filter_map(|p| p.id.as_ref().zip(p.vectors.as_ref()))
                    .filter_map(|(k, v)| match v.vectors_options.as_ref().unwrap() {
                        VectorsOptions::Vector(vector) => Some((k, &vector.data)),
                        _ => None,
                    })
                    .filter_map(|(k, v)| match k.point_id_options.as_ref() {
                        Some(PointIdOptions::Num(id)) => Some((format!("{id}"), v)),
                        Some(PointIdOptions::Uuid(id)) => Some((id.to_string(), v)),
                        _ => None,
                    })
                    .collect();

                // Maybe we should just set it from here instead of doing an info query
                let embed_dims = point_vecs[0].1.len();

                assert!(point_vecs.iter().all(|(_, v)| v.len() == embed_dims));

                let hash_to_uuid = points_to_hover_lookup(&point_vecs);

                if let Ok(mut app_state) = app_lock.lock() {
                    app_state.hash_to_uuid = hash_to_uuid;
                }

                let df = points_to_dataframe(embed_dims, point_vecs);

                dbg!(&df);

                rt.spawn_blocking({
                    let task_count = task_count.clone();
                    task_count.fetch_add(1, Ordering::Relaxed);

                    move || {
                        let df_proj = project_embeddings(umap_lock, df);

                        if let Ok(mut app_state) = app_lock.lock() {
                            app_state.umap_df = df_proj;
                        } else {
                            log::warn!("Could not access app state");
                        }

                        task_count.fetch_sub(1, Ordering::Relaxed);
                    }
                });
            }

            task_count.fetch_sub(1, Ordering::Relaxed);
        });
    }

    fn trigger_semantic_query(&self) {
        let rt = self.rt.handle().to_owned();
        let app_state = self.app_state.clone();
        let qdclient = self.qdclient.clone();
        let task_count = self.task_count.clone();
        let umap_lock = self.umap.clone();

        let (collection_name, model_id, query_string) = if let Ok(app_state) = self.app_state.lock()
        {
            (
                app_state.collection_name.clone(),
                app_state.semantic.embed_model.clone(),
                app_state.semantic.text.clone(),
            )
        } else {
            return;
        };

        if query_string.is_empty() {
            log::info!("No query");
            return;
        }
        if collection_name.is_none() {
            log::info!("No collection selected. Skipping");
            return;
        }

        log::info!("Running query with {model_id}");

        let collection_name = collection_name.unwrap();

        self.rt.handle().spawn(async move {
            task_count.fetch_add(2, Ordering::Relaxed);

            // Perform the embedding in a background thread, since CPU/GPU-bound
            let embedding = rt
                .spawn_blocking({
                    let task_count = task_count.clone();
                    move || {
                        let model = TextEmbedding::try_new(
                            fastembed::InitOptions::new(model_id).with_show_download_progress(true),
                        )
                        .unwrap();

                        let mut embeddings = model.embed(vec![&query_string], None).unwrap();

                        if embeddings.len() != 1 {
                            log::error!("Expected only one embedding for text:\n{query_string}");
                            task_count.fetch_sub(2, Ordering::Relaxed);
                            return None;
                        }

                        embeddings.pop()
                    }
                })
                .await
                .ok()
                .flatten();

            if embedding.is_none() {
                return;
            }

            let embedding = embedding.unwrap();

            // map embedding to a point and display in a background thread
            rt.spawn_blocking({
                let app_state = app_state.clone();
                let embedding = embedding.clone();
                let task_count = task_count.clone();

                move || {
                    let x_u = if let Ok(umap_guard) = umap_lock.lock()
                        && let Some(umap) = umap_guard.as_ref()
                    {
                        Python::with_gil(|py| {
                            let umap = umap.bind(py);
                            let x_u = umap.call_method1("transform", (vec![&embedding],)).unwrap();
                            // TODO extract result to query_point
                            let x_u: Vec<[f32; 2]> = x_u.extract()?;

                            Ok::<_, PyErr>(x_u)
                        })
                        .ok()
                    } else {
                        None
                    };

                    if let Some(x) = x_u.and_then(|mut it| it.pop())
                        && let Ok(mut app_state) = app_state.lock()
                    {
                        // This doesn't trigger a UI redraw.
                        // It's actually the spinner animation instead of data changes.
                        app_state.semantic.query_point = Some((x[0].to_f64(), x[1].to_f64()));
                    }

                    task_count.fetch_sub(1, Ordering::Relaxed);
                }
            });

            // Continue async coro by querying Qdrant to get n_neighbors
            let resp = qdclient
                .query(
                    QueryPointsBuilder::new(collection_name.as_str())
                        .query(embedding.clone())
                        .limit(10),
                )
                .await
                .unwrap();

            // Stringify ids of neighbors
            let matched_ids = resp
                .result
                .iter()
                .map(
                    |pv| match pv.id.as_ref().unwrap().point_id_options.as_ref().unwrap() {
                        PointIdOptions::Num(id) => (format!("{id}"), pv.score),
                        PointIdOptions::Uuid(id) => (id.to_string(), pv.score),
                    },
                )
                .collect::<BTreeMap<_, _>>();

            if let Ok(mut app_state) = app_state.lock() {
                app_state.semantic.matched_ids = Arc::new(matched_ids);
            }
            task_count.fetch_sub(1, Ordering::Relaxed);
        });
    }

    fn render_inspector(&mut self, ui: &mut egui::Ui) -> anyhow::Result<()> {
        let embed_dims = if let Ok(app_state) = self.app_state.lock() {
            app_state.embed_dims
        } else {
            0
        };

        egui::SidePanel::left("Left panel")
            // .resizable(false)
            .default_width(300.0)
            .show_inside(ui, |ui| {
                Frame::new().inner_margin(8.0).show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_enabled_ui(self.task_count.load(Ordering::Relaxed) < 1, |ui| {
                            if ui.button("Refresh").clicked() {
                                self.refresh_points();
                            }
                        });
                    });
                });

                // ui.separator();

                ui.vertical_centered_justified(|ui| {
                    ui.group(|ui| {
                        ui.vertical_centered(|ui| {
                            ui.heading("Query");
                        });

                        let want_semantic_query = ui
                            .add_enabled_ui(self.task_count.load(Ordering::Relaxed) < 1, |ui| {
                                let mut app_state = self.app_state.lock().unwrap();
                                let semantic = &mut app_state.semantic;

                                if !is_valid_embedding(embed_dims, semantic.embed_model.clone()) {
                                    semantic.embed_model = valid_embeddings(embed_dims)
                                        .first()
                                        .map(|m| m.model.clone())
                                        .unwrap();
                                }

                                let start_query = semantic.text.clone();
                                let start_model = semantic.embed_model.clone();

                                let display_model =
                                    TextEmbedding::get_model_info(&semantic.embed_model)
                                        .map(|it| it.model_code.as_str())
                                        .unwrap_or_default();

                                egui::ComboBox::from_id_salt("embed_model")
                                    .selected_text(display_model)
                                    .width(ui.available_width())
                                    .truncate()
                                    .show_ui(ui, |ui| {
                                        for model in valid_embeddings(embed_dims).iter() {
                                            ui.selectable_value(
                                                &mut semantic.embed_model,
                                                model.model.clone(),
                                                model.model_code.as_str(),
                                            )
                                            .on_hover_text(model.description.as_str());
                                        }
                                    });

                                start_model != semantic.embed_model
                                    || ui.input(|i| {
                                        i.modifiers.ctrl && i.key_pressed(egui::Key::Enter)
                                    })
                                    || (ui.text_edit_multiline(&mut semantic.text).lost_focus()
                                        && start_query != semantic.text)
                            })
                            .inner;

                        if want_semantic_query {
                            self.trigger_semantic_query();
                        }

                        // Grid does not honor justification
                        // TODO: try the table in egui_extras instead
                        egui::Grid::new("semantic_matches")
                            .num_columns(2)
                            .striped(true)
                            .show(ui, |ui| {
                                let mut app_state = self.app_state.lock().unwrap();
                                let matched_ids = app_state.semantic.matched_ids.clone();
                                let selected = &mut app_state.select_point;

                                let matched_ids = matched_ids
                                    .iter()
                                    .sorted_by(|(_, v0), (_, v1)| v1.total_cmp(v0));

                                for (id, score) in matched_ids {
                                    // Instead of truncating during resize, this is forcing the minimum
                                    // width to the size of the UUID + scores. None of the other
                                    // techniques below help.
                                    ui.selectable_value(selected, Some(id.clone()), id.clone());

                                    // let job = LayoutJob::simple_singleline(
                                    //     id.to_string(),
                                    //     TextStyle::Button.resolve(ui.style()),
                                    //     Color32::GREEN, // How to get current text color from ui?
                                    // );

                                    // Still doesn't truncate
                                    // let mut job = LayoutJob::default();
                                    // let format = TextFormat {
                                    //     font_id: TextStyle::Button.resolve(ui.style()),
                                    //     ..Default::default()
                                    // };
                                    // job.append(id, 0.0, format);
                                    // job.wrap = TextWrapping {
                                    //     max_rows: 1,
                                    //     break_anywhere: true,
                                    //     ..Default::default()
                                    // };

                                    // Truncates but at minimum size instead of available width
                                    // ui.style_mut().wrap_mode = Some(TextWrapMode::Truncate);

                                    // ui.selectable_value(selected, Some(id.clone()), job);

                                    // Expanded versions do no better...
                                    // let label =
                                    //     egui::SelectableLabel::new(selected.as_ref() == Some(id), job);
                                    //
                                    // if ui.add(label).clicked() {
                                    //     *selected = Some(id.clone());
                                    // }
                                    // if ui
                                    //     .selectable_label(hover_point.as_ref() == Some(id), job)
                                    //     .clicked()
                                    // {
                                    //     app_state.hover_point = Some(id.clone());
                                    // }

                                    // This is the only variant that truncates properly in this context,
                                    // but then we lose selectability.
                                    // ui.add(egui::Label::new(id).truncate());

                                    ui.label(score.to_string());
                                    ui.end_row();
                                }
                            });
                    });
                });

                ui.group(|ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("Details");
                    });

                    ScrollArea::vertical().show(ui, |ui| {
                        ui.vertical(|ui| {
                            let app_state = self.app_state.lock().unwrap();
                            for (k, v) in &app_state.point_details {
                                CollapsingHeader::new(k).default_open(v.len() < 128).show(
                                    ui,
                                    |ui| {
                                        ui.label(v);
                                    },
                                );
                            }

                            // Add an extra line to prevent clipping on long text
                            let font_id = egui::TextStyle::Body.resolve(ui.style());
                            ui.add_space(font_id.size / 2.0);
                        });
                    });
                });
            });
        Ok(())
    }

    fn render_plot(&mut self, ui: &mut egui::Ui) -> anyhow::Result<()> {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let PlotResponse {
                hovered_plot_item, ..
            } = Plot::new("My Plot")
                // .height(500.0)
                // .legend(Legend::default())
                .show(ui, |plot_ui| {
                    let (proj_df, select_point, details_id, matched_ids) = {
                        let app_state = self.app_state.lock().unwrap();
                        (
                            app_state.umap_df.clone(),
                            app_state.select_point.clone(),
                            app_state.point_details.get("id").cloned(),
                            app_state.semantic.matched_ids.clone(),
                        )
                    };

                    let uuid = proj_df["uuid"].str().unwrap();
                    let x0 = extract_f64(&proj_df, "umap0").unwrap();
                    let x1 = extract_f64(&proj_df, "umap1").unwrap();

                    izip!(uuid.iter(), x0.iter(), x1.iter())
                        .filter_map(|(uuid, x0, x1)| optzip!(uuid, x0, x1))
                        .for_each(|(uuid, x0, x1)| {
                            let id = uuid.to_string();
                            let name = uuid.to_string();

                            let match_score = matched_ids.get(&id);
                            let is_detail = details_id.as_ref().map(|v| v == &id).unwrap_or(false);
                            let is_select =
                                select_point.as_ref().map(|v| v == &id).unwrap_or(false);

                            let radius = match true {
                                _ if is_select => 8.0,
                                _ if is_detail => 5.0,
                                _ => 3.0,
                            };

                            let shape = if is_select {
                                MarkerShape::Diamond
                            } else {
                                MarkerShape::Circle
                            };

                            let alpha = match true {
                                _ if is_detail => 255,
                                _ if match_score.is_some() => 196,
                                _ => 128,
                            };

                            let color = if let Some(score) = match_score {
                                PALETTE.eval_continuous(score.to_f64())
                            } else {
                                PALETTE.eval_continuous(0.0)
                            };
                            let color =
                                Color32::from_rgba_unmultiplied(color.r, color.g, color.b, alpha);
                            let points = Points::new(name.clone(), vec![[x0, x1]])
                                .id(id.clone())
                                .shape(shape)
                                .radius(radius)
                                .filled(true)
                                .color(color);

                            plot_ui.points(points);
                        });
                    if let Ok(app_state) = self.app_state.lock()
                        && let Some((x, y)) = &app_state.semantic.query_point
                    {
                        plot_ui.points(
                            Points::new("Query", vec![[*x, *y]])
                                .shape(MarkerShape::Cross)
                                .radius(10.0)
                                .color(Color32::RED),
                        )
                    }
                });

            let refresh_point = {
                let mut app_state = self.app_state.lock().unwrap();
                let hovered_id = hovered_plot_item
                    .and_then(|h| (app_state.hash_to_uuid.get(&h)))
                    .cloned();

                hovered_id
                    .as_ref()
                    .and_then(|uuid| (app_state.hover_point.replace(uuid.clone())));

                if ui.input(|i| i.pointer.primary_clicked()) {
                    let old_id = hovered_id
                        .as_ref()
                        .and_then(|uuid| (app_state.select_point.replace(uuid.clone())));

                    if old_id == app_state.select_point {
                        app_state.select_point = None;
                    }
                }

                let details_id = app_state
                    .select_point
                    .as_ref()
                    .or(app_state.hover_point.as_ref())
                    .cloned();

                if app_state.point_details.get("id") != details_id.as_ref()
                    && let Some(id) = details_id.as_ref()
                {
                    app_state.point_details.clear();
                    app_state.point_details.insert("id".into(), id.clone());
                    details_id
                } else {
                    None
                }
            };

            if let Some(uuid) = refresh_point {
                let rt = self.rt.handle().to_owned();
                let app_state = self.app_state.clone();
                let qdclient = self.qdclient.clone();
                let task_count = self.task_count.clone();

                let collection_name = self
                    .app_state
                    .lock()
                    .ok()
                    .and_then(|s| s.collection_name.clone())
                    .unwrap();

                rt.spawn(async move {
                    task_count.fetch_add(1, Ordering::Relaxed);
                    let resp = qdclient
                        .get_points(
                            GetPointsBuilder::new(collection_name.as_str(), vec![uuid.into()])
                                .with_payload(true),
                        )
                        .await
                        .unwrap();

                    if let Some(point) = resp.result.first()
                        && let Ok(mut app_state) = app_state.lock()
                    {
                        app_state
                            .point_details
                            .extend(point.payload.iter().map(|(k, v)| {
                                (
                                    k.clone(),
                                    v.as_str().cloned().unwrap_or_else(|| v.to_string()),
                                )
                            }));
                    }
                    task_count.fetch_sub(1, Ordering::Relaxed);
                });
            }
        });
        Ok(())
    }
}

async fn refresh_collection_info(app_state: Arc<Mutex<AppState>>, qdclient: Arc<Qdrant>) {
    let selected_collection = app_state
        .lock()
        .ok()
        .and_then(|s| s.collection_name.clone());

    if let Some(name) = selected_collection
        && let Ok(info) = qdclient.collection_info(name).await
    {
        log::info!("Collection info: {info:?}");

        // Okay, this is just a tad ridiculous...
        if let Ok(mut app_state) = app_state.lock()
            && let Some(vectors_config::Config::Params(params)) = info
                .result
                .and_then(|i| i.config)
                .and_then(|i| i.params)
                .and_then(|i| i.vectors_config)
                .and_then(|i| i.config)
        {
            app_state.embed_dims = dbg!(params.size as usize);
        }
    }
}

/// Filters available embedding models by a vector size
#[cached]
fn valid_embeddings(embed_dims: usize) -> Arc<Vec<fastembed::ModelInfo<EmbeddingModel>>> {
    let all_embeddings = TextEmbedding::list_supported_models();

    Arc::new(
        all_embeddings
            .iter()
            .filter(|m| m.dim == embed_dims)
            .sorted_by(|m, n| {
                m.model_code
                    .cmp(&n.model_code)
                    .then_with(|| format!("{m:?}").cmp(&format!("{n:?}")))
            })
            .cloned()
            .collect::<Vec<_>>(),
    )
}

#[cached]
fn is_valid_embedding(embed_dims: usize, embedding_model: EmbeddingModel) -> bool {
    valid_embeddings(embed_dims)
        .iter()
        .map(|m| m.model.clone())
        .contains(&embedding_model)
}

/// Create a lookup table of `egui::Ids` to UUIDs for determining which entry has mouse focus.
fn points_to_hover_lookup(point_vecs: &Vec<(String, &Vec<f32>)>) -> HashMap<egui::Id, String> {
    point_vecs
        .iter()
        .flat_map(|(id, _)| {
            // Ehh???!?! egui::Id hashes to different values for &str vs String
            [
                (egui::Id::new(id.to_string()), id.to_string()),
                (egui::Id::new(id), id.to_string()),
            ]
        })
        .collect()
}

/// Project embeddings from a DataFrame into 2-D coordinates using UMAP.
///
/// If `umap` is Some, then the existing mapping will be used to transform the embeddings.
/// Otherwise, a new mapping will be initialized and fitted to the given data.
///
/// # Arguments:
/// * `umap` - maybe a handle to a UMAP instance
/// * `df` - a DataFrame where each column is an embedding dimension
/// # Returns
/// * A new DataFrame containing the points project onto a 2-D plane
fn project_embeddings(umap: Arc<Mutex<Option<Py<PyAny>>>>, df: DataFrame) -> DataFrame {
    let x_umap = Python::with_gil(|py| {
        let df = df.drop("uuid").unwrap();

        let mut umap_guard = umap.lock().unwrap();

        let umap = match umap_guard.as_ref() {
            Some(umap) => {
                log::info!("Reusing existing umap instance");
                umap.bind(py).clone()
            }
            _ => {
                log::info!("Fitting new umap instance");
                let umap = UMAP
                    .bind(py)
                    .call(
                        (),
                        Some(&pydict! { py;
                            "n_neighbors" => *UMAP_NEIGHBORS
                        }),
                    )
                    .unwrap();

                let umap = umap
                    .call_method1("fit", (PyDataFrame(df.clone()),))
                    .unwrap();

                *umap_guard = Some(umap.clone().unbind());
                umap
            }
        };

        let x_u = umap
            .call_method1("transform", (PyDataFrame(df.clone()),))
            .unwrap();

        let x_u: Vec<[f32; 2]> = x_u.extract()?;

        Ok::<_, PyErr>(x_u)
    })
    .unwrap();

    let (umap0, umap1): (Vec<_>, Vec<_>) = x_umap.into_iter().map(|[a, b]| (a, b)).unzip();

    // dbg!((&df_proj,));
    df! {
        "uuid" => df.column("uuid").unwrap().as_series().unwrap(),
        "umap0" => umap0,
        "umap1" => umap1,
    }
    .unwrap()
}

/// Explodes embeddings from arrays into DataFrame columns
fn points_to_dataframe(embed_dims: usize, point_vecs: Vec<(String, &Vec<f32>)>) -> DataFrame {
    let mut df = DataFrame::default();

    df.with_column(Series::new(
        "uuid".into(),
        point_vecs
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>(),
    ))
    .unwrap();

    for i in 0..embed_dims {
        let ser = Series::new(
            format!("x{i:04}").into(),
            point_vecs.iter().map(|(_, v)| v[i]).collect::<Vec<_>>(),
        );

        df.with_column(ser).unwrap();
    }
    df
}

impl eframe::App for MyEguiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::TopBottomPanel::top("Header").show_inside(ui, |ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::TOP), |ui| {
                    Frame::new().inner_margin(8.0).show(ui, |ui| {
                        // ui.horizontal(|ui| {
                        // ui.label(RichText::new("Collection").heading().strong());
                        let enabled = self.task_count.load(Ordering::Relaxed) < 1;
                        let (collections, start_value) = {
                            let app_state = self.app_state.lock().unwrap();
                            (
                                app_state.available_collections.clone(),
                                app_state.collection_name.clone(),
                            )
                        };

                        let mut dummy = start_value.clone();

                        ui.add_enabled_ui(enabled, |ui| {
                            let resp = egui::ComboBox::from_label("Collection")
                                .selected_text(
                                    RichText::new(start_value.as_ref().unwrap_or(&"".to_string()))
                                        .strong(),
                                )
                                // .width(ui.available_width())
                                // .truncate()
                                .show_ui(ui, |ui| {
                                    for name in collections.iter() {
                                        ui.selectable_value(
                                            &mut dummy,
                                            Some(name.to_string()),
                                            name,
                                        );
                                    }

                                    start_value != dummy
                                });

                            if resp.inner.unwrap_or(false) {
                                if let Ok(mut app_state) = self.app_state.lock() {
                                    *app_state = AppState::new();
                                    app_state.collection_name = dummy;
                                }

                                if let Ok(mut umap) = self.umap.lock() {
                                    *umap = None;
                                }

                                self.refresh_points();
                                self.refresh_collections();
                            }
                        });
                    });
                    // });
                });
            });

            self.render_inspector(ui).unwrap();
            self.render_plot(ui).unwrap();
        });
        egui::TopBottomPanel::bottom("Footer").show(ctx, |ui| {
            ui.columns(3, |cols| {
                // TODO: Settings modal dialog
                cols[0].horizontal(|_| {
                    // if ui.button("⚙").clicked() {
                    //     log::warn!("Not implemented!");
                    // }
                });
                cols[1].horizontal(|ui| {
                    // TODO: only calculate this on change
                    let num_points = self.app_state.lock().ok().map(|s| s.umap_df.height());
                    if let Some(count) = num_points {
                        ui.label(format!("{count} points"));
                    }
                });
                cols[2].with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if self.task_count.load(Ordering::Relaxed) > 0 {
                        ui.spinner();
                        ui.label("Loading");
                    } else {
                        let builder = UiBuilder::new()
                            .id_salt("ready_refresh_widget")
                            .sense(Sense::click());
                        let scoped = ui.scope_builder(builder, |ui| {
                            let size = egui::Vec2::splat(18.0);
                            let (response, painter) = ui.allocate_painter(size, Sense::hover());
                            let rect = response.rect;
                            painter.circle_filled(
                                rect.center(),
                                6.0,
                                Color32::from_rgb(100, 200, 100),
                            );
                        });

                        if scoped.response.clicked() {
                            self.refresh_points();
                        }

                        scoped.response.on_hover_text("Refresh data");

                        ui.label("Ready");
                    }
                });
            });
        });
    }
}

fn extract_f64(df: &DataFrame, colname: &str) -> Result<Float64Chunked> {
    Ok(df
        .column(colname)?
        .cast(&DataType::Float64)?
        .f64()?
        .to_owned())
}
