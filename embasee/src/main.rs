use eframe::egui;
use egui::RichText;
use qdrant_client::qdrant::ScrollPointsBuilder;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use tokio::{runtime::Runtime, sync::mpsc, task};
use tonic::Code;

use qdrant_client::{
    Qdrant,
    qdrant::{CreateCollectionBuilder, Distance, QueryPointsBuilder, VectorParamsBuilder},
};

use emberlain::CodeSnippet;

const LLM_MODEL: &str = "devstral:latest";
const LLM_BASE_URL: &str = "http://10.10.10.100:11434";
const EMBEDDING_DIMS: u64 = 384;
const COLLECTION_NAME: &str = "my_collection";

fn main() -> anyhow::Result<()> {
    // let rt = tokio::runtime::Builder::new_multi_thread()
    //     .enable_all()
    //     .build()?;
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
#[derive(Default, Debug, Clone)]
struct AppState {
    // selected_pet: Option,
    // pets: Vec,
    // pet_image: Option,
    add_form: AddForm,
    snippet: Option<CodeSnippet>,
}

#[derive(Default, Debug, Clone)]
struct AddForm {
    show: bool,
    name: String,
    age: String,
    kind: String,
}

struct MyEguiApp {
    running: AtomicBool,
    rt: Runtime,
    qdclient: Arc<Qdrant>,
    app_state: Arc<Mutex<AppState>>,
}

impl MyEguiApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Customize egui here with cc.egui_ctx.set_fonts and cc.egui_ctx.set_visuals.
        // Restore app state using cc.storage (requires the "persistence" feature).
        // Use the cc.gl (a glow::Context) to create graphics shaders and buffers that you can use
        // for e.g. egui::PaintCallback.

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .unwrap();

        // TODO: periodic timer to fetch ids. Get vectors for new ids. Update clusters.
        let qdclient = Arc::new(Qdrant::from_url("http://localhost:6334").build().unwrap());
        let client = qdclient.clone();
        rt.spawn(async move {
            if !client
                .collection_exists(COLLECTION_NAME)
                .await
                .expect("Qdrant client error")
            {
                client
                    .create_collection(
                        CreateCollectionBuilder::new(COLLECTION_NAME).vectors_config(
                            VectorParamsBuilder::new(EMBEDDING_DIMS, Distance::Cosine),
                        ),
                    )
                    .await
                    .unwrap();
            }
        });

        Self {
            running: AtomicBool::new(true),
            rt,
            qdclient,
            app_state: Arc::new(Mutex::new(Default::default())),
        }
    }
}

impl MyEguiApp {
    fn render_inspector(&mut self, ui: &mut egui::Ui) -> anyhow::Result<()> {
        egui::SidePanel::left("Left panel")
            // .resizable(false)
            .default_width(300.0)
            .show_inside(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.label("Info!");
                    ui.separator();
                    ui.vertical(|ui| {
                        egui::Grid::new("something").num_columns(2).show(ui, |ui| {
                            let mut app_state = self.app_state.lock().unwrap();
                            ui.label("name:");
                            ui.text_edit_singleline(&mut app_state.add_form.name);
                            ui.end_row();
                            ui.label("age");
                            ui.text_edit_singleline(&mut app_state.add_form.age);
                            ui.end_row();
                            ui.label("kind");
                            ui.text_edit_singleline(&mut app_state.add_form.kind);
                            ui.end_row();
                        });
                        if ui.button("Do it").clicked() {
                            println!("Doing it");
                            let rt = self.rt.handle().to_owned();
                            let app_lock = self.app_state.clone();
                            let qdclient = self.qdclient.clone();

                            // std::thread::spawn(move || {
                            // rt.block_on(async move {
                            rt.spawn(async move {
                                let query = QueryPointsBuilder::new(COLLECTION_NAME)
                                    .limit(5)
                                    .with_payload(true);
                                // let resp = qdclient.query(query);
                                println!("Spawned it");
                                let resp = qdclient
                                    .scroll(
                                        ScrollPointsBuilder::new(COLLECTION_NAME)
                                            .limit(2)
                                            .with_payload(true)
                                            .with_vectors(true),
                                    )
                                    .await;
                                println!("Found {} results", resp.as_ref().unwrap().result.len());

                                let entry = &resp.as_ref().unwrap().result[0];
                                let payload = &entry.payload;
                                // dbg!(Vec::from_iter(payload.values().map(|v| v.clone().as_str().unwrap().to_owned())));
                                dbg!(serde_json::to_string_pretty(payload).unwrap());

                                let thing = serde_json::to_value(payload).unwrap();

                                // Value::from(&resp.unwrap().result[0].payload.get_key_value);
                                // dbg!(resp.unwrap());
                                let mut app_state = app_lock.lock().unwrap();
                                app_state.snippet = serde_json::from_value(thing).ok();
                                app_state.add_form.name = payload
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .cloned()
                                    .unwrap_or("???".into());
                                println!("Done it");

                                let vectors: Vec<_> = resp
                                    .as_ref()
                                    .unwrap()
                                    .result
                                    .iter()
                                    .map(|p| (&p.id, &p.vectors))
                                    .collect();
                                dbg!(&vectors);
                            });
                            // });
                        }
                    });
                    ui.separator();
                });
                ui.vertical(|ui| {
                    let app_state = self.app_state.lock().unwrap();
                    if let Some(snippet) = &app_state.snippet {
                        ui.label(&snippet.summary);
                    }
                });
            });
        Ok(())
    }
    fn render_plot(&mut self, ui: &mut egui::Ui) -> anyhow::Result<()> {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.spacing_mut().item_spacing.x = 0.0;
                ui.label("Immediate mode is a GUI paradigm that lets you create a GUI with less code and simpler control flow. For example, this is how you create a ");
                let _ = ui.small_button("button");
                ui.label(" in egui:");
            });
        });
        Ok(())
    }
}

impl eframe::App for MyEguiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::TopBottomPanel::top("Upper").show_inside(ui, |ui| {
                ui.vertical_centered(|ui| {
                    ui.heading(RichText::new("Hello World!").size(24.0).strong());
                });
            });
            self.render_inspector(ui).unwrap();
            self.render_plot(ui).unwrap();
        });
    }
}
